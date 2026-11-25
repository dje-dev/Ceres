#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Provenance

/*
  lc0ex -- the executable format, its precompiled kernels, the builder that produces them and the
  reference CUDA runtime -- is the work of Kovax (GitHub: Kovax007) for Leela Chess Zero.

  This file is a C# port of parts of that runtime:
    src/neural/backends/lc0ex-cuda/runtime/lc0ex_cuda.cc     (kernel launch tables, null-pointer parameters,
                                                              CUDA graph construction, launching)
    src/neural/backends/lc0ex-cuda/network_lc0ex_cuda.cc     (input upload, output download, padding)

  Original sources: https://github.com/Kovax007/lc0, branch tcec-season-30-swiss-lc0ex, commit 29ec4b44.
  Copyright (C) 2026 The LCZero Authors. Licensed under the GNU General Public License, version 3 or
  (at your option) any later version, with the additional permission for NVIDIA libraries stated there.

  Ported to C# and modified for Ceres in November 2026.
  Differences from the original: a batch is covered by several program launches chosen from measured
  launch times (the original pads to the next program, with an optional two-way split), all launches
  of a batch are awaited once, and the wait blocks on an event.
*/

#endregion

#region Using directives

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;

using ManagedCuda;
using ManagedCuda.BasicTypes;

#endregion

namespace Ceres.Chess.NNBackends.LC0EX
{
  /// <summary>
  /// One execution context for an LC0EXExecutable: a non-blocking stream, an execution
  /// allocation (sized for the largest program used), pinned host input/output buffers,
  /// precomputed kernel launch tables and one instantiated CUDA graph per program.
  ///
  /// Not thread-safe: the owner (one evaluator instance) serializes calls to Run.
  ///
  /// Usage per batch: write rows 0..n-1 of InputMasks/InputValues (Lc0 112-plane
  /// packed format), call Run(n), then read rows 0..n-1 of the Output* spans.
  ///
  /// Batch planning: an lc0ex program computes a fixed number of positions, so a batch which
  /// falls between two program sizes would waste the difference if simply padded up to the
  /// next program (e.g. 272 positions run on a 512 program cost as much as 512).
  /// Instead each batch is covered by a sequence of program launches ("chunks") chosen to
  /// minimize the total measured launch time (see ComputePlan); only the last chunk is padded.
  /// All chunks of a batch are queued on the stream and awaited with a single synchronization.
  /// Programs which no plan uses are not instantiated, which also keeps the execution
  /// allocation as small as the largest program actually worth running.
  /// </summary>
  public sealed unsafe class LC0EXExecution : IDisposable
  {
    public const int NUM_PLANES = 112;
    public const int NUM_POLICY = 1858;

    const string IN_MASKS = "/input/plane_masks";
    const string IN_VALUES = "/input/plane_values";
    const string OUT_POLICY = "/output/policy";
    const string OUT_WDL = "/output/wdl";
    const string OUT_MLH = "/output/mlh";

    /// <summary>
    /// The shared executable (modules, weights) this execution runs.
    /// </summary>
    public LC0EXExecutable Executable { get; }

    /// <summary>
    /// Largest number of positions accepted by Run.
    /// </summary>
    public int MaxBatchSize { get; }

    /// <summary>
    /// Batch size of the largest program this execution uses.
    /// </summary>
    public int MaxProgramBatchSize { get; private set; }

    /// <summary>
    /// Batch sizes of the programs this execution uses (ascending).
    /// Programs of the executable which no batch plan uses are omitted.
    /// </summary>
    public int[] ProgramBatchSizes { get; private set; }

    /// <summary>
    /// If every program used produces a moves-left output.
    /// </summary>
    public bool HasMovesLeft { get; private set; }

    /// <summary>
    /// If kernels are launched via instantiated CUDA graphs (otherwise one launch per node).
    /// </summary>
    public bool UseGraphs { get; }

    /// <summary>
    /// If completion is awaited by blocking on a CUDA event (the waiting thread sleeps)
    /// rather than by cuStreamSynchronize (which, under the default primary context
    /// scheduling policy, spins a CPU core for the whole duration of the batch).
    /// </summary>
    public bool UseBlockingSync { get; }

    public Span<ulong> InputMasks => new((void*)pinnedMasks, MaxBatchSize * NUM_PLANES);
    public Span<float> InputValues => new((void*)pinnedValues, MaxBatchSize * NUM_PLANES);
    public ReadOnlySpan<float> OutputPolicyLogits => new((void*)pinnedPolicy, MaxBatchSize * NUM_POLICY);
    public ReadOnlySpan<float> OutputWDLLogits => new((void*)pinnedWDL, MaxBatchSize * 3);
    public ReadOnlySpan<float> OutputMLH => new((void*)pinnedMLH, MaxBatchSize);


    sealed class ProgramState
    {
      public LC0EXProgramDef Program;
      public int BatchSize;
      public ulong MasksOffset, ValuesOffset, PolicyOffset, WDLOffset, MLHOffset;
      public bool HasMLH;
      public CUfunction[] NodeFunctions;
      public IntPtr[][] NodeKernelParams;  // per node: managed copy of the void*[] (for direct launches)
      public IntPtr NativeKernelParams;    // per node void*[] arrays, contiguous (for graph nodes)
      public int[] NodeParamOffset;        // index of node's first slot in NativeKernelParams
      public CUgraphExec GraphExec;
      public bool GraphValid;
    }

    /// <summary>
    /// Number of timed launches per program when measuring its latency (the minimum is used).
    /// </summary>
    const int NUM_COST_MEASUREMENTS = 3;

    /// <summary>
    /// Relative cost penalty per step of the program ladder applied when planning batches,
    /// so that near-equal programs resolve to the smaller one (see ComputePlan).
    /// </summary>
    const double PLAN_SIZE_BIAS = 0.01;

    ProgramState[] programs;
    int[] planProgram;   // [n] = index (into programs) of the next chunk to launch when n positions remain
    int[] planCapacity;  // [n] = total position slots computed for a batch of n (sum of the chunk program sizes)
    CUstream stream;
    CUevent doneEvent;
    CUdeviceptr execRaw;
    ulong execBase;
    ulong execSize;
    IntPtr pinnedMasks, pinnedValues, pinnedPolicy, pinnedWDL, pinnedMLH;
    IntPtr argValueStorage;  // 8 bytes per kernel parameter slot, all programs
    bool disposed;
    int disposeStarted;


    /// <summary>
    /// Creates an execution for an already acquired executable; takes ownership of one reference
    /// (released on Dispose, or here if construction fails).
    /// </summary>
    /// <param name="executable">the shared executable (modules and weights)</param>
    /// <param name="maxBatchSize">largest batch that will be passed to Run</param>
    /// <param name="useGraphs">if programs are launched as instantiated CUDA graphs</param>
    /// <param name="useBlockingSync">if completion is awaited by blocking on an event instead of spinning</param>
    public LC0EXExecution(LC0EXExecutable executable, int maxBatchSize, bool useGraphs = true, bool useBlockingSync = true)
    {
      Executable = executable;
      UseGraphs = useGraphs;
      UseBlockingSync = useBlockingSync;
      MaxBatchSize = maxBatchSize;

      try
      {
        if (maxBatchSize < 1)
        {
          throw new ArgumentOutOfRangeException(nameof(maxBatchSize));
        }

        // Candidates: every program up to the smallest one covering maxBatchSize
        // (a larger program could only ever add padding).
        List<LC0EXProgramDef> all = executable.Def.Programs;
        LC0EXProgramDef covering = all.FirstOrDefault(p => p.BatchSize >= maxBatchSize) ?? all[^1];
        LC0EXProgramDef[] candidates = all.Where(p => p.BatchSize <= covering.BatchSize).ToArray();
        int[] sizes = candidates.Select(p => p.BatchSize).ToArray();
        double[] costs = new double[sizes.Length];

        // The launch time of each program is measured once per executable (by the first
        // execution which needs it) and then shared by all executions on the device.
        lock (executable.ProgramCostLock)
        {
          if (!sizes.All(executable.ProgramCostsMs.ContainsKey))
          {
            Build(candidates);
            MeasureMissingProgramCosts();
          }

          for (int i = 0; i < sizes.Length; i++)
          {
            costs[i] = executable.ProgramCostsMs[sizes[i]];
          }
        }

        ComputePlan(sizes, costs, maxBatchSize, out int[] choice, out planCapacity);

        // Keep only the programs some plan actually launches.
        bool[] isUsed = new bool[sizes.Length];
        for (int n = 1; n <= maxBatchSize; n++)
        {
          isUsed[choice[n]] = true;
        }

        LC0EXProgramDef[] used = candidates.Where((p, i) => isUsed[i]).ToArray();
        if (programs == null || programs.Length != used.Length)
        {
          if (programs != null)
          {
            FreeResources();
          }
          Build(used);
        }

        int[] usedIndex = new int[sizes.Length];
        for (int i = 0, next = 0; i < sizes.Length; i++)
        {
          usedIndex[i] = isUsed[i] ? next++ : -1;
        }

        planProgram = new int[maxBatchSize + 1];
        for (int n = 1; n <= maxBatchSize; n++)
        {
          planProgram[n] = usedIndex[choice[n]];
        }
      }
      catch
      {
        disposeStarted = 1;
        FreeResources();
        disposed = true;
        executable.Release();
        throw;
      }
    }


    /// <summary>
    /// Allocates all device and host resources for the specified programs
    /// (which become the programs of this execution).
    /// </summary>
    void Build(LC0EXProgramDef[] used)
    {
      ProgramState[] states = new ProgramState[used.Length];
      for (int i = 0; i < used.Length; i++)
      {
        states[i] = BindIO(used[i]);
      }

      programs = states;
      ProgramBatchSizes = used.Select(p => p.BatchSize).ToArray();
      MaxProgramBatchSize = used[^1].BatchSize;
      HasMovesLeft = states.All(p => p.HasMLH);

      Executable.Device.GraphCaptureRWLock.EnterWriteLock();
      try
      {
        Executable.Device.SetCurrent();
        AllocateResources(used);

        int firstSlot = 0;
        foreach (ProgramState ps in programs)
        {
          firstSlot = BuildLaunchTables(ps, firstSlot);
        }

        if (UseGraphs)
        {
          foreach (ProgramState ps in programs)
          {
            BuildGraph(ps);
          }
        }
      }
      finally
      {
        Executable.Device.GraphCaptureRWLock.ExitWriteLock();
      }
    }


    #region Batch planning

    /// <summary>
    /// Measures the wall-clock time of one launch (upload, kernels, download) of every program
    /// of this execution not yet measured for the executable. Caller holds ProgramCostLock.
    /// The time does not depend on the input values, so the (zeroed) input buffers are used as is.
    /// </summary>
    void MeasureMissingProgramCosts()
    {
      foreach (ProgramState ps in programs)
      {
        if (Executable.ProgramCostsMs.ContainsKey(ps.BatchSize))
        {
          continue;
        }

        // Download no more rows than the output buffers hold.
        int count = Math.Min(ps.BatchSize, MaxBatchSize);
        double bestMs = double.MaxValue;
        for (int i = 0; i <= NUM_COST_MEASUREMENTS; i++)
        {
          long start = Stopwatch.GetTimestamp();
          Executable.Device.GraphCaptureRWLock.EnterReadLock();
          try
          {
            Executable.Device.SetCurrent();
            EnqueueChunk(ps, 0, count);
            WaitForCompletion();
          }
          finally
          {
            Executable.Device.GraphCaptureRWLock.ExitReadLock();
          }

          // The first launch is a warmup and is not counted.
          if (i > 0)
          {
            bestMs = Math.Min(bestMs, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
          }
        }

        Executable.ProgramCostsMs[ps.BatchSize] = bestMs;
      }
    }


    /// <summary>
    /// Computes, for every batch size n in [1, maxBatch], the cheapest way to cover n positions
    /// with program launches, given the batch size and measured cost of each available program.
    /// (Dynamic program over the number of positions remaining; any program may be used any
    /// number of times, and a program larger than the remainder ends the plan, padded.)
    ///
    /// Programs often cost almost exactly the same per position, so that which of them wins
    /// would be decided by measurement noise. A larger program is therefore used only if it is
    /// faster by more than PLAN_SIZE_BIAS per step of the program ladder; otherwise the smaller
    /// one is preferred (same throughput, less execution memory, same plans in every run).
    /// </summary>
    /// <param name="sizes">program batch sizes (ascending)</param>
    /// <param name="costs">cost of one launch of each program</param>
    /// <param name="maxBatch">largest batch size to plan for</param>
    /// <param name="choice">[n] = index of the program to launch first when n positions remain</param>
    /// <param name="capacity">[n] = sum of the program sizes launched for a batch of n</param>
    internal static void ComputePlan(int[] sizes, double[] costs, int maxBatch, out int[] choice, out int[] capacity)
    {
      choice = new int[maxBatch + 1];
      capacity = new int[maxBatch + 1];
      double[] best = new double[maxBatch + 1];

      double[] biasedCosts = new double[costs.Length];
      for (int k = 0; k < costs.Length; k++)
      {
        biasedCosts[k] = costs[k] * (1 + PLAN_SIZE_BIAS * k);
      }

      for (int n = 1; n <= maxBatch; n++)
      {
        double bestCost = double.MaxValue;
        int bestProgram = 0;

        // Ascending, so equal costs resolve to the smaller program.
        for (int k = 0; k < sizes.Length; k++)
        {
          double cost = biasedCosts[k] + best[Math.Max(0, n - sizes[k])];
          if (cost < bestCost)
          {
            bestCost = cost;
            bestProgram = k;
          }
        }

        best[n] = bestCost;
        choice[n] = bestProgram;
        capacity[n] = sizes[bestProgram] + capacity[Math.Max(0, n - sizes[bestProgram])];
      }
    }


    /// <summary>
    /// Returns the total number of position slots computed for a batch of numPositions
    /// (the sum of the sizes of the programs launched). The slots beyond numPositions are padding.
    /// </summary>
    public int PaddedCapacity(int numPositions)
    {
      if (numPositions < 0 || numPositions > MaxBatchSize)
      {
        throw new ArgumentOutOfRangeException(nameof(numPositions), $"{numPositions} not in [0, {MaxBatchSize}].");
      }
      return planCapacity[numPositions];
    }


    /// <summary>
    /// Returns the batch sizes of the programs launched (in order) for a batch of numPositions.
    /// Diagnostic use (allocates).
    /// </summary>
    public int[] PlanFor(int numPositions)
    {
      if (numPositions < 1 || numPositions > MaxBatchSize)
      {
        throw new ArgumentOutOfRangeException(nameof(numPositions), $"{numPositions} not in [1, {MaxBatchSize}].");
      }

      List<int> chunks = new();
      for (int remaining = numPositions; remaining > 0;)
      {
        int size = programs[planProgram[remaining]].BatchSize;
        chunks.Add(size);
        remaining -= size;
      }
      return chunks.ToArray();
    }


    /// <summary>
    /// Returns the measured latency in milliseconds of one launch of the program with the specified batch size.
    /// </summary>
    public double ProgramCostMs(int programBatchSize)
    {
      lock (Executable.ProgramCostLock)
      {
        return Executable.ProgramCostsMs[programBatchSize];
      }
    }

    #endregion


    #region Setup

    ProgramState BindIO(LC0EXProgramDef p)
    {
      ProgramState ps = new() { Program = p, BatchSize = p.BatchSize };
      int b = p.BatchSize;

      ulong Require(string name, LC0EXDataType dt, int cols, bool optional, out bool found)
      {
        found = p.Buffers.TryGetValue(name, out LC0EXBufferDef buf);
        if (!found)
        {
          if (optional)
          {
            return 0;
          }
          throw new FormatException($"lc0ex program {p.Name} has no buffer '{name}'.");
        }

        bool shapeOK = buf.Shape.Length == 2 && buf.Shape[0] == (ulong)b && buf.Shape[1] == (ulong)cols;
        if (buf.DataType != dt || !shapeOK || !buf.IsRowMajor)
        {
          throw new FormatException($"lc0ex program {p.Name} buffer '{name}' is {buf.DataType}[{string.Join(",", buf.Shape)}], "
                                  + $"expected row-major {dt}[{b},{cols}].");
        }
        if (buf.Offset + buf.SizeBytes > p.ExecutionSizeBytes)
        {
          throw new FormatException($"lc0ex program {p.Name} buffer '{name}' extends past the execution allocation.");
        }
        return buf.Offset;
      }

      ps.MasksOffset = Require(IN_MASKS, LC0EXDataType.U64, NUM_PLANES, false, out _);
      ps.ValuesOffset = Require(IN_VALUES, LC0EXDataType.F32, NUM_PLANES, false, out _);
      ps.PolicyOffset = Require(OUT_POLICY, LC0EXDataType.F32, NUM_POLICY, false, out _);
      ps.WDLOffset = Require(OUT_WDL, LC0EXDataType.F32, 3, false, out _);
      ps.MLHOffset = Require(OUT_MLH, LC0EXDataType.F32, 1, true, out ps.HasMLH);
      return ps;
    }


    void AllocateResources(LC0EXProgramDef[] used)
    {
      CUResult res = DriverAPINativeMethods.Streams.cuStreamCreate(ref stream, CUStreamFlags.NonBlocking);
      LC0EXExecutable.CheckResult(res, "cuStreamCreate");

      if (UseBlockingSync)
      {
        res = DriverAPINativeMethods.Events.cuEventCreate(ref doneEvent, CUEventFlags.BlockingSync | CUEventFlags.DisableTiming);
        LC0EXExecutable.CheckResult(res, "cuEventCreate");
      }

      execSize = used.Max(p => p.ExecutionSizeBytes);
      ulong align = Math.Max(1UL, used.Max(p => p.ExecutionAlignment));
      if (execSize > 0)
      {
        res = DriverAPINativeMethods.MemoryManagement.cuMemAlloc_v2(ref execRaw, execSize + align - 1);
        LC0EXExecutable.CheckResult(res, $"cuMemAlloc({execSize} bytes execution)");
        execBase = ((ulong)execRaw.Pointer + (align - 1)) & ~(align - 1);
      }

      // Inputs hold a whole batch plus room to pad its last chunk up to the largest program;
      // outputs hold a whole batch (only rows of real positions are downloaded).
      ulong inputRows = (ulong)MaxBatchSize + (ulong)MaxProgramBatchSize;
      ulong outputRows = (ulong)MaxBatchSize;
      pinnedMasks = AllocPinned(inputRows * NUM_PLANES * sizeof(ulong));
      pinnedValues = AllocPinned(inputRows * NUM_PLANES * sizeof(float));
      pinnedPolicy = AllocPinned(outputRows * NUM_POLICY * sizeof(float));
      pinnedWDL = AllocPinned(outputRows * 3 * sizeof(float));
      pinnedMLH = AllocPinned(outputRows * sizeof(float));

      long totalSlots = used.Sum(p => p.Nodes.Sum(n => (long)Executable.Def.Kernels[n.KernelIndex].Parameters.Length));
      argValueStorage = (IntPtr)NativeMemory.AllocZeroed((nuint)Math.Max(1, totalSlots * sizeof(ulong)));
    }


    static IntPtr AllocPinned(ulong bytes)
    {
      IntPtr p = IntPtr.Zero;
      CUResult res = DriverAPINativeMethods.MemoryManagement.cuMemHostAlloc(ref p, bytes, CUMemHostAllocFlags.Portable);
      LC0EXExecutable.CheckResult(res, $"cuMemHostAlloc({bytes})");
      NativeMemory.Clear((void*)p, (nuint)bytes);
      return p;
    }


    /// <summary>
    /// Resolves every kernel parameter slot of every node into a device value stored in
    /// argValueStorage, and builds the void*[] kernelParams array pointing at those values.
    /// NULL_POINTER slots get a zero value and do not consume a Node.Argument.
    /// Returns the index of the first slot of argValueStorage following this program's.
    /// </summary>
    int BuildLaunchTables(ProgramState ps, int firstSlot)
    {
      LC0EXProgramDef p = ps.Program;
      int numNodes = p.Nodes.Length;
      ps.NodeFunctions = new CUfunction[numNodes];
      ps.NodeKernelParams = new IntPtr[numNodes][];
      ps.NodeParamOffset = new int[numNodes];

      int slots = p.Nodes.Sum(n => Executable.Def.Kernels[n.KernelIndex].Parameters.Length);
      ps.NativeKernelParams = (IntPtr)NativeMemory.AllocZeroed((nuint)Math.Max(1, slots * IntPtr.Size));
      IntPtr* kernelParams = (IntPtr*)ps.NativeKernelParams;
      ulong* values = (ulong*)argValueStorage + firstSlot;

      int slot = 0;
      for (int n = 0; n < numNodes; n++)
      {
        LC0EXNodeDef node = p.Nodes[n];
        LC0EXKernelDef kernel = Executable.Def.Kernels[node.KernelIndex];
        ps.NodeFunctions[n] = Executable.Functions[node.KernelIndex];
        ps.NodeParamOffset[n] = slot;

        int argIndex = 0;
        IntPtr[] managed = new IntPtr[kernel.Parameters.Length];
        for (int i = 0; i < kernel.Parameters.Length; i++)
        {
          ulong value = 0;
          LC0EXParameterType type = kernel.Parameters[i];
          if (type != LC0EXParameterType.NullPointer)
          {
            LC0EXArgumentDef arg = node.Arguments[argIndex++];
            if (type == LC0EXParameterType.U32 && arg.Kind != LC0EXArgumentKind.Parameter)
            {
              throw new FormatException($"lc0ex node {n} ({kernel.Function}): U32 parameter bound to a non-parameter argument.");
            }

            value = arg.Kind switch
            {
              LC0EXArgumentKind.Allocation when arg.AllocationKind == LC0EXAllocationKind.Persistent => Executable.PersistentBase + arg.AllocationOffset,
              LC0EXArgumentKind.Allocation => execBase + arg.AllocationOffset,
              LC0EXArgumentKind.Symbol => Executable.SymbolAddresses[(arg.SymbolBinaryIndex, arg.SymbolName)],
              _ => throw new NotSupportedException($"lc0ex runtime parameter argument '{arg.ParameterName}' not supported.")
            };
          }

          ulong* valueSlot = values + slot;
          *valueSlot = value;
          kernelParams[slot] = (IntPtr)valueSlot;
          managed[i] = (IntPtr)valueSlot;
          slot++;
        }

        ps.NodeKernelParams[n] = managed;
      }

      return firstSlot + slot;
    }


    /// <summary>
    /// Builds an explicit CUDA graph (one kernel node per program node, edges from the
    /// node dependencies), instantiates and uploads it. On failure the program falls back
    /// to direct launches.
    /// </summary>
    void BuildGraph(ProgramState ps)
    {
      LC0EXProgramDef p = ps.Program;
      CUgraph graph = default;
      CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphCreate(ref graph, 0);
      LC0EXExecutable.CheckResult(res, "cuGraphCreate");

      try
      {
        CUgraphNode[] graphNodes = new CUgraphNode[p.Nodes.Length];
        IntPtr* kernelParams = (IntPtr*)ps.NativeKernelParams;
        for (int n = 0; n < p.Nodes.Length; n++)
        {
          LC0EXNodeDef node = p.Nodes[n];
          CUgraphNode[] deps = new CUgraphNode[node.Dependencies.Length];
          for (int d = 0; d < deps.Length; d++)
          {
            deps[d] = graphNodes[node.Dependencies[d]];
          }

          CudaKernelNodeParams kp = new()
          {
            func = ps.NodeFunctions[n],
            gridDimX = node.Grid[0],
            gridDimY = node.Grid[1],
            gridDimZ = node.Grid[2],
            blockDimX = node.Block[0],
            blockDimY = node.Block[1],
            blockDimZ = node.Block[2],
            sharedMemBytes = node.DynamicSharedMemoryBytes,
            kernelParams = (IntPtr)(kernelParams + ps.NodeParamOffset[n]),
            extra = IntPtr.Zero,
          };

          res = DriverAPINativeMethods.GraphManagment.cuGraphAddKernelNode(ref graphNodes[n], graph, deps, deps.Length, ref kp);
          LC0EXExecutable.CheckResult(res, $"cuGraphAddKernelNode(program {p.Name}, node {n})");
        }

        res = DriverAPINativeMethods.GraphManagment.cuGraphInstantiate(ref ps.GraphExec, graph, CUgraphInstantiate_flags.None);
        LC0EXExecutable.CheckResult(res, $"cuGraphInstantiate(program {p.Name})");
        res = DriverAPINativeMethods.GraphManagment.cuGraphUpload(ps.GraphExec, stream);
        LC0EXExecutable.CheckResult(res, $"cuGraphUpload(program {p.Name})");
        ps.GraphValid = true;
      }
      catch (Exception exc)
      {
        Console.WriteLine($"Warning: lc0ex CUDA graph for program {p.Name} could not be built, using direct launches. {exc.Message}");
        if (ps.GraphExec.Pointer != IntPtr.Zero)
        {
          DriverAPINativeMethods.GraphManagment.cuGraphExecDestroy(ps.GraphExec);
          ps.GraphExec = default;
        }
        ps.GraphValid = false;
      }
      finally
      {
        DriverAPINativeMethods.GraphManagment.cuGraphDestroy(graph);
      }
    }

    #endregion


    /// <summary>
    /// Evaluates rows 0..numPositions-1 of the input buffers into rows 0..numPositions-1 of the
    /// output buffers, as one or more program launches (see class remarks). The unused rows of
    /// the last program launched are padded with copies of the last position (as lc0 does).
    /// Returns the total number of position slots computed (same as PaddedCapacity(numPositions)).
    /// </summary>
    public int Run(int numPositions)
    {
      if (disposed)
      {
        throw new ObjectDisposedException(nameof(LC0EXExecution));
      }
      if (numPositions < 1 || numPositions > MaxBatchSize)
      {
        throw new ArgumentOutOfRangeException(nameof(numPositions), $"{numPositions} not in [1, {MaxBatchSize}].");
      }

      Executable.Device.GraphCaptureRWLock.EnterReadLock();
      try
      {
        Executable.Device.SetCurrent();

        // Queue every chunk on the stream (which executes them in order), then wait once.
        int offset = 0;
        int remaining = numPositions;
        while (remaining > 0)
        {
          ProgramState ps = programs[planProgram[remaining]];
          int count = Math.Min(ps.BatchSize, remaining);
          EnqueueChunk(ps, offset, count);
          offset += count;
          remaining -= count;
        }

        WaitForCompletion();
      }
      finally
      {
        Executable.Device.GraphCaptureRWLock.ExitReadLock();
      }

      return planCapacity[numPositions];
    }


    /// <summary>
    /// Queues on the stream one launch of a program for the count positions starting at
    /// row firstRow of the input buffers: input upload, kernels, download of count output rows
    /// to rows starting at firstRow of the output buffers. If count is less than the program's
    /// batch size the remaining rows are padded with copies of the last position; such a
    /// chunk must be the last one of its batch (the padding is written after its rows).
    /// The context must be current and the graph capture read lock held.
    /// </summary>
    void EnqueueChunk(ProgramState ps, int firstRow, int count)
    {
      int b = ps.BatchSize;
      ulong* masks = (ulong*)pinnedMasks + (long)firstRow * NUM_PLANES;
      float* values = (float*)pinnedValues + (long)firstRow * NUM_PLANES;

      if (count < b)
      {
        ReadOnlySpan<ulong> lastMasks = new(masks + (long)(count - 1) * NUM_PLANES, NUM_PLANES);
        ReadOnlySpan<float> lastValues = new(values + (long)(count - 1) * NUM_PLANES, NUM_PLANES);
        for (int row = count; row < b; row++)
        {
          lastMasks.CopyTo(new Span<ulong>(masks + (long)row * NUM_PLANES, NUM_PLANES));
          lastValues.CopyTo(new Span<float>(values + (long)row * NUM_PLANES, NUM_PLANES));
        }
      }

      Check(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(new CUdeviceptr(execBase + ps.MasksOffset), (IntPtr)masks,
                                                                             (ulong)b * NUM_PLANES * sizeof(ulong), stream), "HtoD masks");
      Check(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(new CUdeviceptr(execBase + ps.ValuesOffset), (IntPtr)values,
                                                                             (ulong)b * NUM_PLANES * sizeof(float), stream), "HtoD values");

      if (UseGraphs && ps.GraphValid)
      {
        Check(DriverAPINativeMethods.GraphManagment.cuGraphLaunch(ps.GraphExec, stream), "cuGraphLaunch");
      }
      else
      {
        LC0EXNodeDef[] nodes = ps.Program.Nodes;
        for (int n = 0; n < nodes.Length; n++)
        {
          LC0EXNodeDef node = nodes[n];
          CUResult res = DriverAPINativeMethods.Launch.cuLaunchKernel(ps.NodeFunctions[n],
                                                                      node.Grid[0], node.Grid[1], node.Grid[2],
                                                                      node.Block[0], node.Block[1], node.Block[2],
                                                                      node.DynamicSharedMemoryBytes, stream,
                                                                      ps.NodeKernelParams[n], null);
          if (res != CUResult.Success)
          {
            // (Message built only on failure: this loop runs once per node per batch.)
            Check(res, $"cuLaunchKernel(node {n}, {Executable.Def.Kernels[node.KernelIndex].Function})");
          }
        }
      }

      Check(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2((IntPtr)((float*)pinnedPolicy + (long)firstRow * NUM_POLICY),
                                                                             new CUdeviceptr(execBase + ps.PolicyOffset),
                                                                             (ulong)count * NUM_POLICY * sizeof(float), stream), "DtoH policy");
      Check(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2((IntPtr)((float*)pinnedWDL + (long)firstRow * 3),
                                                                             new CUdeviceptr(execBase + ps.WDLOffset),
                                                                             (ulong)count * 3 * sizeof(float), stream), "DtoH wdl");
      if (ps.HasMLH)
      {
        Check(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2((IntPtr)((float*)pinnedMLH + firstRow),
                                                                               new CUdeviceptr(execBase + ps.MLHOffset),
                                                                               (ulong)count * sizeof(float), stream), "DtoH mlh");
      }
    }


    /// <summary>
    /// Waits until everything queued on the stream has completed.
    /// </summary>
    void WaitForCompletion()
    {
      if (UseBlockingSync)
      {
        Check(DriverAPINativeMethods.Events.cuEventRecord(doneEvent, stream), "cuEventRecord");
        Check(DriverAPINativeMethods.Events.cuEventSynchronize(doneEvent), "cuEventSynchronize");
      }
      else
      {
        Check(DriverAPINativeMethods.Streams.cuStreamSynchronize(stream), "cuStreamSynchronize");
      }
    }


    static void Check(CUResult res, string what) => LC0EXExecutable.CheckResult(res, what);


    #region Dispose

    void FreeResources()
    {
      try
      {
        Executable.Device.SetCurrent();
      }
      catch
      {
        // Ignore; best effort cleanup.
      }

      if (programs != null)
      {
        foreach (ProgramState ps in programs)
        {
          if (ps == null)
          {
            continue;
          }
          if (ps.GraphExec.Pointer != IntPtr.Zero)
          {
            DriverAPINativeMethods.GraphManagment.cuGraphExecDestroy(ps.GraphExec);
            ps.GraphExec = default;
          }
          if (ps.NativeKernelParams != IntPtr.Zero)
          {
            NativeMemory.Free((void*)ps.NativeKernelParams);
            ps.NativeKernelParams = IntPtr.Zero;
          }
        }
      }

      if (stream.Pointer != IntPtr.Zero)
      {
        // Let anything still queued (e.g. after a failed Run) finish before its buffers are freed.
        DriverAPINativeMethods.Streams.cuStreamSynchronize(stream);
      }

      if (doneEvent.Pointer != IntPtr.Zero)
      {
        DriverAPINativeMethods.Events.cuEventDestroy_v2(doneEvent);
        doneEvent = default;
      }

      if (stream.Pointer != IntPtr.Zero)
      {
        DriverAPINativeMethods.Streams.cuStreamDestroy_v2(stream);
        stream = default;
      }

      if (execRaw.Pointer != 0)
      {
        DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(execRaw);
        execRaw = default;
      }

      foreach (IntPtr p in new[] { pinnedMasks, pinnedValues, pinnedPolicy, pinnedWDL, pinnedMLH })
      {
        if (p != IntPtr.Zero)
        {
          DriverAPINativeMethods.MemoryManagement.cuMemFreeHost(p);
        }
      }
      pinnedMasks = pinnedValues = pinnedPolicy = pinnedWDL = pinnedMLH = IntPtr.Zero;

      if (argValueStorage != IntPtr.Zero)
      {
        NativeMemory.Free((void*)argValueStorage);
        argValueStorage = IntPtr.Zero;
      }

      programs = null;
    }


    public void Dispose()
    {
      // (Safe against being called more than once, also concurrently: the reference is released once.)
      if (Interlocked.Exchange(ref disposeStarted, 1) == 0)
      {
        disposed = true;
        FreeResources();
        Executable.Release();
      }
    }

    #endregion


    public override string ToString()
      => $"<LC0EXExecution {Executable} using programs [{string.Join(",", ProgramBatchSizes ?? Array.Empty<int>())}] graphs={UseGraphs}>";
  }
}
