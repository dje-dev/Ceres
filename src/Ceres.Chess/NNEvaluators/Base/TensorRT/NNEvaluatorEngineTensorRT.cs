#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security;

using Ceres.Base.Math;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;

using Ceres.Chess;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NNEvaluators.Internals;
using Ceres.Chess.LC0.Batches;

#endregion

namespace Chess.Ceres.NNEvaluators.TensorRT
{
  /// <summary>
  /// Implementation of NNEvaluator which uses the NVIDIA TensorRT library.
  /// </summary>
  public class NNEvaluatorEngineTensorRT : NNEvaluator
  {
    const int MAX_SESSIONS = 16; // hardcoded constant in C++ code

    const int NUM_TOPK_POLICY = 64;// set in C++ code

    public override bool IsWDL => Config.IsWDL;
    public override bool HasM => Config.HasM;


    public readonly NNEvaluatorEngineTensorRTConfig Config;

    public readonly int SessionID;
    //    public readonly bool ShouldRebuild;
    public readonly int GPUID;
    public readonly string UFFFN;
    public readonly List<Position> CalibPositions;
    public readonly bool RetrieveValueFCActivations;

    public readonly NNEvaluatorPrecision Precision;
    public readonly bool Shared;

    public readonly NNEvaluatorEngineTensorRTConfig.NetTypeEnum Type;
    public readonly NNEvaluatorEngineTensorRTConfig.TRTPriorityLevel PriorityLevel;

    // TODO: possibly the requirement of Moves can be lifted?
    public override InputTypes InputsRequired => InputTypes.Boards | InputTypes.Moves;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="engineID"></param>
    /// <param name="uffFN"></param>
    /// <param name="isWDL"></param>
    /// <param name="hasM"></param>
    /// <param name="gpuID"></param>
    /// <param name="type"></param>
    /// <param name="batchSize"></param>
    /// <param name="precision"></param>
    /// <param name="priorityLevel"></param>
    /// <param name="calibPositions"></param>
    /// <param name="retrieveValueFCActivations"></param>
    /// <param name="shared"></param>
    public NNEvaluatorEngineTensorRT(string engineID, string uffFN, bool isWDL, bool hasM, int gpuID, 
                                     NNEvaluatorEngineTensorRTConfig.NetTypeEnum type,
                                    int batchSize, NNEvaluatorPrecision precision,
                                    NNEvaluatorEngineTensorRTConfig.TRTPriorityLevel priorityLevel = NNEvaluatorEngineTensorRTConfig.TRTPriorityLevel.High,
                                    List<Position> calibPositions = null, bool retrieveValueFCActivations = false,
                                    bool shared = true)
    {
      Config = new NNEvaluatorEngineTensorRTConfig(uffFN, type == NNEvaluatorEngineTensorRTConfig.NetTypeEnum.Ceres ? "TRT_DJE" : "TRT_LZ0", 
                                           batchSize, precision, gpuID, isWDL, hasM, priorityLevel, retrieveValueFCActivations);

      SessionID = GetSessionToAttachTo(Config, shared);
      UFFFN = uffFN;
      GPUID = gpuID;
      EngineNetworkID = engineID;
      CalibPositions = calibPositions;
      Precision = precision;
      PriorityLevel = priorityLevel;
      Shared = shared;
      Type = type;

      //      ShouldRebuild = shouldRebuild;
      RetrieveValueFCActivations = retrieveValueFCActivations;
    }


    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat batch, bool retrieveSupplementalResults = false) // ** MAKE VIRTUAL
    {
      if (RetrieveValueFCActivations != retrieveSupplementalResults) throw new Exception("Value of parameter " + retrieveSupplementalResults + " does not math constructor configuration");
      return EvaluateBatch(batch, batch.NumPos, false, retrieveValueFCActivations: retrieveSupplementalResults);
    }


    static volatile int batchCount = 0;
    static object trtLockObj = new object();
    
    public PositionEvaluationBatch EvaluateBatch(IEncodedPositionBatchFlat batch, int numToProcess, bool verbose = false,
                                           bool retrieveValueFCActivations = false) // ** MAKE VIRTUAL
    {
      // Serialize access since executor does not support parallel operations
      lock (sessionActiveLocks[SessionID])
      {
        return DoEvaluateBatch(batch, numToProcess, verbose, retrieveValueFCActivations);
      }
    }

    public static int rawInputModificationIndex = -1;
    public static float rawInputModificationDelta = 0;

    // If not padded to 4 we sometimes get many warnings logged to console (each batch)
    // about the chosen algorithm not being suitable.
    // TODO: revisit, can/should we turn this padding off in at least some cases?
    const int PADDING_ALIGN = 4;

    PositionEvaluationBatch DoEvaluateBatch(IEncodedPositionBatchFlat batch, int numToProcess, bool verbose = false, 
                                      bool retrieveValueFCActivations = false) // ** MAKE VIRTUAL
    {
      if (batch.NumPos > Config.MaxBatchSize)
      {
        throw new Exception($"Requested batch size {batch.NumPos} for TensorRT exceeds specified maximum {Config.MaxBatchSize}");
      }

      if (batch.NumPos == 0)
      {
        throw new Exception("Empty batch");
      }

      NNEvaluatorStats.UpdateStatsForBatch(GPUID, numToProcess);

      if (numToProcess <= 0) throw new ArgumentOutOfRangeException($"numToProcess must be greater than zero {numToProcess}");
      if (numToProcess > 2048) throw new ArgumentOutOfRangeException("TensorRT engines are unlikely to be able to process >2048 positions");
      //LZTrainingPositionServerBatch batchCalib = null; // no longer used LZTrainingPositionServerBatch.GenBatchFromPositions(CalibPositions);

      int numToProcessPadded = (int)MathUtils.RoundedUp(numToProcess, PADDING_ALIGN);

      //TimingStats stats = new TimingStats();
      //using (new TimingBlock(stats, TimingBlock.LoggingType.None)) slow
      {
        float[] floatsCalib = null;// batchCalib.EncodedPosExpandedAsFloats;
        float[] lz0FloatsCalib = null;// ChessNetTFExecutor.RebuildInputsForLZ0Network(floatsCalib, numToProcess);

        const bool USE_TOP_K = false;// this was disabled Oct 2021 because became buggy
        Span<float> rawResultsPolicy =
          USE_TOP_K ? stackalloc float[Config.MaxBatchSize * NUM_TOPK_POLICY * 2] // one 4 byte entry for each index (as int 8), one 4 byte entry for each probability
                    : new float[numToProcessPadded * EncodedPolicyVector.POLICY_VECTOR_LENGTH]; 
        int NUM_VALUE_OUTPUTS = (Config.IsWDL ? 3 : 1);
        Span<FP16> results = stackalloc FP16[numToProcessPadded * NUM_VALUE_OUTPUTS];
        Span<FP16> resultsMLH = stackalloc FP16[numToProcessPadded * 1];

        if (retrieveValueFCActivations) throw new Exception("The ONNX version of our TensorRT library does not expose inner layers");
        float[] rawResultsConvValFlat = retrieveValueFCActivations ? new float[numToProcessPadded * NUM_VALUE_OUTPUTS * 32 * 64] : new float[1]; // Note: can't be null or empty, since we use in fixed statement below

#if NOTES
        NOTE: cudaMallocHost was 1.5 slower than just cudaAlloc for the buffers[]
        ** GREAT CUDA SITE:  https://jhui.github.io/2017/03/06/CUDA/

        This can definitely be optimized greatly. To handle small networks and >150,000 nps, we'll need a lot of work:
          - (MAYBE NOT NEEDED) use HALF instead of float as what we pass to network (somehow need to insert preprocessing into ONNX file?)
          - do all of this without multiple memory copies (let C# use pinned buffers directly for initialization)

Updated notes:
          After various optimizations (for 256x20 and 64x6 networks) some good results
            Raw CUDA calculation (plus copy results back): 63k/410k 
            End-to-end from C# calculations:               54k/270k 
          
        Also note that with the smaller network, 64x6 is only slightly faster than 256x20. And besides, we can't keep up with 410k/sec on the LC0 side.
#endif
        TimingStats timeStats = new TimingStats();

        if (false)
        {
          Console.WriteLine("\r\n");
          ((EncodedPositionBatchFlat) batch).DumpPlanes(0);
        }

        //if (batch.PosPlaneValues.Length < )
        //Console.WriteLine("MBS_IS " + MaxBatchSize);

        unsafe
        {
          float* rawResultsValue = stackalloc float[numToProcessPadded * NUM_VALUE_OUTPUTS];
          float* rawResultsMLH = stackalloc float[numToProcessPadded * 1];

          fixed (byte* inputPlaneValuesF = &batch.PosPlaneValues[0])
          fixed (ulong* inputPlaneBitmapsF = &batch.PosPlaneBitmaps[0])

          fixed (float* resultsPolicyPtr = &rawResultsPolicy[0])
          fixed (float* resultsValueFC2Ptr = &rawResultsConvValFlat[0])
          {
            if (verbose) Console.WriteLine($"build with precision { Config.Precision} and batch size {Config.MaxBatchSize } ");

            double bestTime = int.MaxValue;
            using (new TimingBlock($"Session={SessionID} process {numToProcess} " + rawResultsValue[0], timeStats, verbose ? TimingBlock.LoggingType.Console : TimingBlock.LoggingType.None))
            {
              int errCode = 0;
              //            Console.WriteLine($"enter Session={SessionID} process {numToProcess} " + rawResultsValue[0], timeStats, verbose ? TimingBlock.LoggingType.Console : TimingBlock.LoggingType.Console);

              errCode = TRTRun(SessionID,
                               inputPlaneBitmapsF, inputPlaneValuesF, numToProcessPadded,
                               rawInputModificationIndex,
                               rawInputModificationDelta,
                               rawResultsValue,
                               rawResultsMLH,
                               resultsPolicyPtr, resultsValueFC2Ptr);

              if (errCode != 0)
              {
                throw new Exception("Error in TRTRun: " + errCode);
              }

            }

            if (verbose) Console.WriteLine($"\r\n {numToProcess} BEST NUM BATCHES PER SECOND, exit now " + (1.0 / bestTime)); //           System.Environment.Exit(3);

            Span<float> rawResultsValueSpan = new Span<float>(rawResultsValue, numToProcess * NUM_VALUE_OUTPUTS);
            Span<float> rawResultsMLHSpan = new Span<float>(rawResultsMLH, numToProcess * 1);

            if (NUM_VALUE_OUTPUTS == 1)
            {
              for (int i = 0; i < numToProcess; i++)
              {
                results[i] = (FP16)rawResultsValueSpan[i];
                resultsMLH[i] = (FP16)rawResultsMLHSpan[i];
              }
            }
            else if (NUM_VALUE_OUTPUTS == 3)
            {
              for (int i = 0; i < numToProcess; i++)
              {
                int vBase = i * 3;
                results[vBase] = (FP16)rawResultsValueSpan[vBase];
                results[vBase+1] = (FP16)rawResultsValueSpan[vBase+1];
                results[vBase+2] = (FP16)rawResultsValueSpan[vBase+2];

                resultsMLH[i] = (FP16)rawResultsMLHSpan[i];
              }
            }
            else
            {
              throw new Exception("Internal error, unexpected value count");
            }
          }
        }


        if (!Config.IsWDL || !Config.HasM)
        {
          throw new Exception("WDL and/or MLH missing, TRT backend currently probably assumes they are present");
        }

        PositionEvaluationBatch retBatch;
        const bool VALUES_ARE_LOGISTIC = false; // the values are exponentiated already in the C++ code

        if (USE_TOP_K)
        {
          int NUM_ELEMENTS = Config.MaxBatchSize * NUM_TOPK_POLICY;
          Span<Int32> policyIndicies = MemoryMarshal.Cast<float, Int32>(rawResultsPolicy).Slice(0, NUM_ELEMENTS);
          Span<float> policyProbabilities = rawResultsPolicy.Slice(Config.MaxBatchSize * NUM_TOPK_POLICY, NUM_ELEMENTS);

          retBatch = new PositionEvaluationBatch(Config.IsWDL, Config.HasM,
                                                 numToProcess, results,
                                                 NUM_TOPK_POLICY,
                                                 policyIndicies, policyProbabilities,
                                                 resultsMLH,
                                                 null, //rawResultsConvValFlat,
                                                 VALUES_ARE_LOGISTIC,
                                                 PositionEvaluationBatch.PolicyType.Probabilities, timeStats);
        }
        else
        {
          //          public PositionEvaluationBatch(bool isWDL, bool hasM,
          //                                       int numPos, Span<FP16> valueEvals,
          //                                       float[] policyProbs,
          //                               FP16[] m,
          //                               NNEvaluatorResultActivations[] activations,
          //                                       bool valsAreLogistic, PolicyType probType, bool policyAlreadySorted, TimingStats stats)

          // NOTE: alternative would be to pass in a mask to the GPU, the batch.ValidMovesMasks could be used to help
          // done below instead. batch.MaskIllegalMovesInPolicyArray(rawResultsPolicy);

          retBatch = new PositionEvaluationBatch(Config.IsWDL, Config.HasM,
                                                 numToProcess, results,
                                                 rawResultsPolicy.Slice(0, numToProcess*1858).ToArray(), // Inefficient 
                                                 resultsMLH.ToArray(),
                                                 null, //rawResultsConvValFlat,
                                                 VALUES_ARE_LOGISTIC,
                                                 PositionEvaluationBatch.PolicyType.LogProbabilities, false, batch, timeStats);

        }

        return retBatch;
      }
  }

    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public override int MaxBatchSize => Config.MaxBatchSize;


#region DLL interface

    // TODO: remove hardcoding
//    public const string DLL = @"c:\dev\CeresOther\trt713\TRTEXEC.DLL";
    public const string DLL = @"c:\dev\CeresOther\trt820\TRTEXEC.DLL";

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    [SuppressUnmanagedCodeSecurity()]
    public unsafe static extern int TRTInit(int sessionID, int gpuID, int priorityLevel,
                                        [MarshalAs(UnmanagedType.LPStr)]string uffFileName,
                                        bool isWDL,
                                        bool forceRebuild,
                                        int maxBatchSize,
                                        float* calibData, int calibDataNumInputs, int precision,
                                        bool retrieveInnerValues);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    [SuppressUnmanagedCodeSecurity()]
    public unsafe static extern int TRTRun(int sessionID, 
                                        ulong* inputPlaneBitmaps,
                                        byte* inputPlaneValues,
                                        int inputDataNumInputs,
                                        int rawInputModificationIndex,
                                        float rawInputModificationDelta,
                                        float* resultValue,
                                        float* resultMLH,
                                        float* resultsPolicy,
                                        float* resultsValueFC2);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    [SuppressUnmanagedCodeSecurity()]
    public unsafe static extern float FP16Convert(ushort *fp16Array);

    
#region Disposal

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    [SuppressUnmanagedCodeSecurity()]
    public unsafe static extern void TRTReset(int sessionID);

    bool isShutdown = false;

    public static string PATH_PREPEND;

    static NNEvaluatorEngineTensorRT()
    {
      // TODO: remove hardcoding
      string path = System.Environment.GetEnvironmentVariable("PATH");
      path = PATH_PREPEND = @"c:\dev\CeresOther\trt713;" + path;
      Environment.SetEnvironmentVariable("PATH", path);
    }


    ~NNEvaluatorEngineTensorRT()
    {
      DoShutdown();
    }

    protected override void DoShutdown()
    {
      if (!isShutdown)
      {
        DetachFromSession(SessionID);
        isShutdown = true;
      }
    }

#region Session management

    static object sessionLockObj = new object();
    static NNEvaluatorEngineTensorRTConfig[] sessionConfigs = new NNEvaluatorEngineTensorRTConfig[MAX_SESSIONS];
    static int[] sessionAttachCounts = new int[MAX_SESSIONS];

    static object[] sessionActiveLocks = new object[MAX_SESSIONS];


    public static void ReleaseAllSessions()
    {
      lock (sessionLockObj)
      {
        // Make sure none are active
        for (int i = 0; i < MAX_SESSIONS; i++)
        {
          if (sessionAttachCounts[i] > 0)
          {
            throw new Exception($"Cannot release session {i}, currently active");
          }
        }

        for (int i = 0; i < MAX_SESSIONS; i++)
        {
          if (sessionConfigs[i] != null)
          {
            TRTReset(i);

            sessionAttachCounts[i] = 0;
            sessionConfigs[i] = null;
            sessionActiveLocks[i] = null;
          }
        }
      }
    }


    static void DetachFromSession(int sessionID)
    {
      lock (sessionLockObj)
      {
        // Someday truly release?
        //TRTReset(sessionID);
        //activeSessions[sessionID] = null;
        sessionAttachCounts[sessionID]--;
      }

    }


    static unsafe int GetSessionToAttachTo(NNEvaluatorEngineTensorRTConfig config, bool shared)
    {
      // TRTInit is not threadsafe, so use of lock above is essential 
      lock (sessionLockObj)
      {
        int firstFree = -1;
        for (int i = 0; i < MAX_SESSIONS; i++)
        {
          if (sessionConfigs[i] == null && firstFree == -1)
          {
            firstFree = i;
          }

          if (sessionConfigs[i] != null && sessionConfigs[i].Equals(config))
          {
            // We have a matching session slot
            if (shared || (!shared & sessionAttachCounts[i] == 0))
            {
              sessionAttachCounts[i]++;
              return i;
            }
          }
        }

        if (firstFree == -1) throw new Exception($"Too many WFEvalNetTensorRT sessions active simultaneously, maximum {MAX_SESSIONS}");

        const bool FORCE_REBUILD = false;
        int errCode = TRTInit(firstFree, config.GPUID, (int)config.PriorityLevel, config.UFFFileName, config.IsWDL, FORCE_REBUILD,
                              config.MaxBatchSize, null, config.MaxBatchSize, (int)config.Precision, config.RetrieveValueFCActivations);

        if (errCode != 0)
        {
          throw new Exception("TRTInit returned error code " + errCode);
        }

        sessionAttachCounts[firstFree]++;
        sessionConfigs[firstFree] = config;
        sessionActiveLocks[firstFree] = new object();
        return firstFree;
      }
    }


#endregion

#endregion

#endregion

  }

}
