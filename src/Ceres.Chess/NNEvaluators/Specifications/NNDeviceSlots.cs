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
using System.Globalization;
using System.Linq;
using System.Text;

using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Remote;

#endregion

namespace Ceres.Chess.NNEvaluators.Specifications
{
  /// <summary>
  /// A device slot: a placement on which a concurrent worker (a tournament game thread
  /// or a suite worker) runs its neural network evaluations. A slot is one GPU or a bundle
  /// of GPUs, either local or on a remote host running "Ceres SERVER".
  ///
  /// A slot hosts one stream (concurrent worker) unless a multiplicity above one was specified,
  /// in which case that many streams are deliberately placed on the same device(s).
  /// </summary>
  [Serializable]
  public sealed class NNDeviceSlot
  {
    /// <summary>
    /// The device(s) of the slot. Only the placement related fields are meaningful
    /// (DeviceIndex, RemoteHost, RemotePort and the optional OverrideEngineType).
    /// </summary>
    public readonly NNEvaluatorDeviceDef[] Devices;

    /// <summary>
    /// Number of concurrent streams which the slot was declared to host.
    /// </summary>
    public readonly int Multiplicity;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="devices"></param>
    /// <param name="multiplicity"></param>
    public NNDeviceSlot(NNEvaluatorDeviceDef[] devices, int multiplicity = 1)
    {
      if (devices == null || devices.Length == 0)
      {
        throw new ArgumentException("A device slot must contain at least one device.", nameof(devices));
      }
      ArgumentOutOfRangeException.ThrowIfLessThan(multiplicity, 1);

      Devices = devices;
      Multiplicity = multiplicity;
    }


    /// <summary>
    /// Hostname of the remote evaluation server, or null if the slot is local.
    /// </summary>
    public string RemoteHost => Devices[0].RemoteHost;

    /// <summary>
    /// TCP port of the remote evaluation server (meaningful only if IsRemote).
    /// </summary>
    public int RemotePort => Devices[0].RemotePort;

    /// <summary>
    /// If the slot is on a remote host.
    /// </summary>
    public bool IsRemote => RemoteHost != null;

    /// <summary>
    /// Short description of where the slot lives ("local" or "host:port").
    /// </summary>
    public string HostDescription => IsRemote ? $"{RemoteHost}:{RemotePort}" : "local";


    /// <summary>
    /// Returns the slot in slot specification syntax (without multiplicity), e.g. "GPU:(0,1)@HOP".
    /// </summary>
    public override string ToString()
    {
      string indices = string.Join(",", Devices.Select(d => d.DeviceIndex.ToString(CultureInfo.InvariantCulture)));
      if (Devices.Length > 1)
      {
        indices = "(" + indices + ")";
      }

      string engine = Devices[0].OverrideEngineType != null ? "#" + Devices[0].OverrideEngineType : "";
      string remote = "";
      if (IsRemote)
      {
        remote = "@" + RemoteHost + (RemotePort != NNRemoteProtocol.DEFAULT_PORT ? "/" + RemotePort : "");
      }

      return $"GPU:{indices}{engine}{remote}";
    }
  }


  /// <summary>
  /// One concurrent stream: a worker bound to a device slot.
  /// </summary>
  /// <param name="Slot">the slot on which the stream runs</param>
  /// <param name="SlotIndex">index of the slot within the slot specification</param>
  /// <param name="StreamIndex">index of this stream among the streams placed on the same slot</param>
  public readonly record struct NNDeviceStream(NNDeviceSlot Slot, int SlotIndex, int StreamIndex)
  {
    /// <summary>
    /// Returns a description such as "GPU:3@station" (or "GPU:3@station #2" for a second or later stream).
    /// </summary>
    public override string ToString() => StreamIndex == 0 ? Slot.ToString() : $"{Slot} #{StreamIndex}";

    /// <summary>
    /// Total time (in milliseconds) which the remote evaluators of this stream have so far spent stalled
    /// because their connection to the evaluation server was lost (0 if the stream runs on a local slot).
    /// Sampled before and after a time-controlled game or search this reveals if it was disrupted.
    /// </summary>
    public long RemoteLostMilliseconds
      => Slot.IsRemote
         ? NNEvaluatorRemote.LostMilliseconds(Slot.RemoteHost, Slot.RemotePort,
                                              Slot.Devices.Select(d => d.DeviceIndex).ToArray(), StreamIndex)
         : 0;
  }


  /// <summary>
  /// Parses and manages a device slots specification, which says on which devices
  /// (local and/or remote) a set of concurrent workers should be spread.
  ///
  /// The syntax is that of a device specification string, read differently:
  ///   - each listed GPU is one slot (rather than one device of a single evaluator),
  ///   - parentheses bundle several GPUs into one slot, and
  ///   - "*N" after a GPU or bundle places N concurrent streams on that slot.
  /// An optional "#engine" and "@host[/port]" apply to every slot of their "+" group.
  ///
  /// Examples:
  ///   "GPU:0+GPU:0,1,2@HOP+GPU:0@station"      5 slots: local 0; HOP 0, 1, 2; station 0
  ///   "GPU:0+GPU:(0,1)@HOP+GPU:3@station"      3 slots: local 0; HOP {0,1} bundled; station 3
  ///   "GPU:0+GPU:(0,1)*2@HOP+GPU:3*3@station"  6 streams: local 0; HOP {0,1} twice; station 3 three times
  ///   "GPU:0,1*2,2@HOP"                        4 streams: HOP 0; HOP 1 twice; HOP 2
  ///
  /// The slot determines only WHERE evaluation happens. The network, execution engine,
  /// batch size hints and options remain those of each engine's own NNEvaluatorDef
  /// (see NNEvaluatorDef.TrySetDevices).
  /// </summary>
  [Serializable]
  public sealed class NNDeviceSlots
  {
    /// <summary>
    /// The specification string from which the slots were parsed.
    /// </summary>
    public readonly string SpecificationString;

    /// <summary>
    /// The slots, in the order specified.
    /// </summary>
    public readonly NNDeviceSlot[] Slots;


    /// <summary>
    /// Constructor which parses a slots specification string.
    /// </summary>
    /// <param name="specificationString"></param>
    public NNDeviceSlots(string specificationString)
    {
      if (string.IsNullOrWhiteSpace(specificationString))
      {
        throw new ArgumentException("Device slots specification is empty.", nameof(specificationString));
      }

      SpecificationString = specificationString.Trim();

      List<NNDeviceSlot> slots = new();
      foreach (string group in SpecificationString.Split('+', StringSplitOptions.TrimEntries))
      {
        ParseGroup(group, slots);
      }

      Slots = slots.ToArray();
    }


    /// <summary>
    /// Returns if specified slots specification string is valid.
    /// </summary>
    /// <param name="specificationString"></param>
    /// <returns></returns>
    public static bool IsValid(string specificationString)
    {
      try
      {
        new NNDeviceSlots(specificationString);
        return true;
      }
      catch
      {
        return false;
      }
    }


    /// <summary>
    /// Total number of streams declared (the sum of the multiplicities of the slots).
    /// This is the default level of concurrency.
    /// </summary>
    public int TotalStreams => Slots.Sum(s => s.Multiplicity);

    /// <summary>
    /// If any of the slots is on a remote host.
    /// </summary>
    public bool AnyRemote => Slots.Any(s => s.IsRemote);


    /// <summary>
    /// Returns the streams to be used for a specified number of concurrent workers
    /// (worker i runs on element i).
    ///
    /// Streams are handed out in rounds: every slot receives its first stream before any slot
    /// receives a second one, and so on up to the declared multiplicities.
    /// For "GPU:0+GPU:(0,1)*2@HOP+GPU:3*3@station" the order is:
    ///   local 0, HOP(0,1), station 3, HOP(0,1), station 3, station 3.
    ///
    /// If more workers are requested than streams were declared the same order repeats
    /// (preserving the declared proportions), i.e. the devices are oversubscribed beyond
    /// what was declared; each additional stream still receives its own stream index.
    /// </summary>
    /// <param name="numConcurrent">number of concurrent workers, or null for the declared total</param>
    /// <returns></returns>
    public NNDeviceStream[] StreamOrder(int? numConcurrent = null)
    {
      int numWorkers = numConcurrent ?? TotalStreams;
      ArgumentOutOfRangeException.ThrowIfLessThan(numWorkers, 1, nameof(numConcurrent));

      // Build one full cycle of slot indices, in rounds.
      List<int> cycle = new();
      int maxMultiplicity = Slots.Max(s => s.Multiplicity);
      for (int round = 0; round < maxMultiplicity; round++)
      {
        for (int slotIndex = 0; slotIndex < Slots.Length; slotIndex++)
        {
          if (Slots[slotIndex].Multiplicity > round)
          {
            cycle.Add(slotIndex);
          }
        }
      }

      int[] nextStreamIndexForSlot = new int[Slots.Length];
      NNDeviceStream[] ret = new NNDeviceStream[numWorkers];
      for (int i = 0; i < numWorkers; i++)
      {
        int slotIndex = cycle[i % cycle.Count];
        ret[i] = new NNDeviceStream(Slots[slotIndex], slotIndex, nextStreamIndexForSlot[slotIndex]++);
      }

      return ret;
    }


    /// <summary>
    /// Validates the slots against the engines which will be placed on them
    /// (to be called before any engine is built):
    ///   - throws if a remote slot would be used by an engine which cannot evaluate remotely
    ///     (for example the external LC0 executable, which would silently run on a local GPU),
    ///   - warns if the same physical GPU appears in two different slots, and
    ///   - warns if more concurrent workers are requested than streams were declared.
    /// </summary>
    /// <param name="engineDefs">definitions of the engines to be placed on the slots</param>
    /// <param name="numConcurrent">number of concurrent workers</param>
    /// <param name="warn">called with the text of any warning</param>
    public void Validate(IEnumerable<GameEngineDef> engineDefs, int numConcurrent, Action<string> warn)
    {
      NNDeviceStream[] streams = StreamOrder(numConcurrent);

      if (streams.Any(s => s.Slot.IsRemote))
      {
        foreach (GameEngineDef engineDef in engineDefs)
        {
          NNEvaluatorDef evaluatorDef = engineDef?.GetEvaluatorDef();
          if (evaluatorDef == null || evaluatorDef.DeviceCombo == NNEvaluatorDeviceComboType.Pooled)
          {
            continue; // no evaluator, or shared (pooled) evaluator which is not re-targeted
          }

          if (!engineDef.SupportsRemoteDevices)
          {
            throw new Exception($"Engine {engineDef.ID} ({engineDef.GetType().Name}) cannot evaluate on a remote host, "
                              + $"but the device slots include remote slots: {SpecificationString}. "
                              + "Use only local slots, or an in-process Ceres engine.");
          }
        }
      }

      // Look for the same physical GPU appearing in two different slots
      // (multiple streams on one slot are deliberate and requested using the "*N" syntax).
      Dictionary<(string host, int port, int deviceIndex), int> firstSlotUsingDevice = new();
      for (int slotIndex = 0; slotIndex < Slots.Length; slotIndex++)
      {
        NNDeviceSlot slot = Slots[slotIndex];
        foreach (NNEvaluatorDeviceDef device in slot.Devices)
        {
          (string, int, int) key = (slot.RemoteHost, slot.IsRemote ? slot.RemotePort : 0, device.DeviceIndex);
          if (firstSlotUsingDevice.TryGetValue(key, out int otherSlotIndex))
          {
            if (otherSlotIndex != slotIndex)
            {
              warn?.Invoke($"WARNING: GPU {device.DeviceIndex} on {slot.HostDescription} appears in two device slots "
                         + $"({Slots[otherSlotIndex]} and {slot}), so their streams will share that GPU. "
                         + "Use the *N syntax if several streams are intended to share a slot.");
            }
          }
          else
          {
            firstSlotUsingDevice[key] = slotIndex;
          }
        }
      }

      if (numConcurrent > TotalStreams)
      {
        warn?.Invoke($"WARNING: concurrency ({numConcurrent}) exceeds the {TotalStreams} stream(s) declared by device slots "
                   + $"{SpecificationString}; the slots will be reused cyclically so additional workers share device(s) "
                   + "(deliberate oversubscription; timing comparability may be reduced).");
      }
    }


    /// <summary>
    /// Verifies that a server is listening on every remote host used by the streams
    /// of a specified number of concurrent workers, throwing a single exception
    /// which lists all unreachable hosts if any are not.
    /// </summary>
    /// <param name="numConcurrent">number of concurrent workers</param>
    /// <param name="timeoutMS">connection timeout per host in milliseconds</param>
    public void VerifyRemoteHostsReachable(int numConcurrent, int timeoutMS = 3000)
    {
      NNDeviceStream[] streams = StreamOrder(numConcurrent);

      List<string> failures = new();
      foreach (IGrouping<(string host, int port), NNDeviceStream> hostGroup
               in streams.Where(s => s.Slot.IsRemote).GroupBy(s => (s.Slot.RemoteHost, s.Slot.RemotePort)))
      {
        if (!NNEvaluatorRemote.TryProbeServer(hostGroup.Key.host, hostGroup.Key.port, timeoutMS, out string error))
        {
          failures.Add($"  {hostGroup.Key.host}:{hostGroup.Key.port} (needed by {hostGroup.Count()} stream(s)): {error}");
        }
      }

      if (failures.Count > 0)
      {
        throw new Exception("Remote NN evaluation server(s) not reachable for device slots " + SpecificationString + Environment.NewLine
                          + string.Join(Environment.NewLine, failures) + Environment.NewLine
                          + "Start a server on each such host using: Ceres SERVER [port=N] [maxclients=N]");
      }
    }


    /// <summary>
    /// Returns a multi-line description of which worker runs on which slot.
    /// </summary>
    /// <param name="streams"></param>
    /// <param name="workerLabel">label of the worker index column (e.g. "TH#")</param>
    /// <returns></returns>
    public static string DescribeStreams(NNDeviceStream[] streams, string workerLabel)
    {
      StringBuilder sb = new();
      for (int i = 0; i < streams.Length; i++)
      {
        sb.AppendLine($"  {workerLabel} {i,3}  {streams[i]}");
      }
      return sb.ToString();
    }


    #region Parsing

    /// <summary>
    /// Parses one "+" group (e.g. "GPU:(0,1)*2,2#TensorRTNative@HOP") and appends its slots.
    /// </summary>
    static void ParseGroup(string group, List<NNDeviceSlot> slots)
    {
      const string EXAMPLES = ", valid examples: GPU:0 or GPU:0,1@HOP or GPU:(0,1)*2@HOP+GPU:3*3@station";

      if (!group.StartsWith("GPU:", StringComparison.OrdinalIgnoreCase))
      {
        throw new Exception($"Device slot group '{group}' is expected to begin with 'GPU:'{EXAMPLES}");
      }

      string body = group.Substring(4);
      if (body.Contains(':') || body.Contains('='))
      {
        throw new Exception($"POOLED and shared (=name) evaluators cannot be used in device slots: '{group}'");
      }

      // Split off the suffix ("#engine" and/or "@host[/port]") which applies to all slots of the group.
      int suffixStart = body.IndexOf('#');
      if (suffixStart < 0)
      {
        int lastAt = body.LastIndexOf('@');
        if (lastAt >= 0 && !float.TryParse(body.Substring(lastAt + 1), NumberStyles.Float, CultureInfo.InvariantCulture, out _))
        {
          suffixStart = lastAt;
        }
      }

      string items = suffixStart >= 0 ? body.Substring(0, suffixStart) : body;
      string suffix = suffixStart >= 0 ? body.Substring(suffixStart) : "";

      if (items.Contains('@') || items.Contains('['))
      {
        throw new Exception($"Device weights (@) and batch sizes ([..]) are not supported in device slots "
                          + $"(they remain part of each engine's own device specification): '{group}'");
      }

      foreach (string itemRaw in SplitAtTopLevelCommas(items, group))
      {
        string item = itemRaw.Trim();
        if (item.Length == 0)
        {
          throw new Exception($"Empty device in device slot group '{group}'{EXAMPLES}");
        }

        // Optional multiplicity ("*N").
        int multiplicity = 1;
        int starIndex = item.LastIndexOf('*');
        if (starIndex >= 0)
        {
          if (!int.TryParse(item.Substring(starIndex + 1), NumberStyles.None, CultureInfo.InvariantCulture, out multiplicity)
           || multiplicity < 1)
          {
            throw new Exception($"Invalid multiplicity in device slot '{item}' of '{group}', expected *N with N >= 1");
          }
          item = item.Substring(0, starIndex).Trim();
        }

        // Either a single device index or a parenthesized bundle of device indices.
        string indices;
        if (item.StartsWith('('))
        {
          if (!item.EndsWith(')'))
          {
            throw new Exception($"Unbalanced parentheses in device slot '{itemRaw}' of '{group}'");
          }
          indices = item.Substring(1, item.Length - 2);
        }
        else
        {
          indices = item;
        }

        foreach (string index in indices.Split(','))
        {
          if (!int.TryParse(index.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out _))
          {
            throw new Exception($"Invalid device index '{index}' in device slot '{itemRaw}' of '{group}'{EXAMPLES}");
          }
        }

        // The slot is now an ordinary device specification, parsed in the usual way.
        NNDevicesSpecificationString slotSpec = new("GPU:" + indices.Replace(" ", "") + suffix);
        slots.Add(new NNDeviceSlot(slotSpec.Devices.Select(d => d.Item1).ToArray(), multiplicity));
      }
    }


    /// <summary>
    /// Splits a string at commas which are not inside parentheses.
    /// </summary>
    static List<string> SplitAtTopLevelCommas(string str, string group)
    {
      List<string> ret = new();
      StringBuilder current = new();
      int depth = 0;
      foreach (char c in str)
      {
        if (c == '(')
        {
          depth++;
        }
        else if (c == ')')
        {
          depth--;
        }

        if (depth < 0 || depth > 1)
        {
          throw new Exception($"Unbalanced or nested parentheses in device slot group '{group}'");
        }

        if (c == ',' && depth == 0)
        {
          ret.Add(current.ToString());
          current.Clear();
        }
        else
        {
          current.Append(c);
        }
      }

      if (depth != 0)
      {
        throw new Exception($"Unbalanced parentheses in device slot group '{group}'");
      }

      ret.Add(current.ToString());
      return ret;
    }

    #endregion


    #region Flat device ID pools (the older way of spreading workers over local GPUs)

    /// <summary>
    /// Returns the (absolute) device IDs assigned to a concurrent worker from a flat pool of device IDs.
    /// The pool is partitioned into consecutive chunks (one per worker) of length devicesPerWorker.
    /// If the pool is smaller than the number of workers times devicesPerWorker the IDs wrap around
    /// cyclically, so workers share devices (deliberate oversubscription).
    /// </summary>
    /// <param name="deviceIDs">the pool of device IDs</param>
    /// <param name="workerIndex">index of the concurrent worker</param>
    /// <param name="devicesPerWorker">number of devices each worker's evaluators span</param>
    /// <returns>the device IDs, or null if no device assignment applies</returns>
    public static int[] DeviceSliceForWorker(int[] deviceIDs, int workerIndex, int devicesPerWorker)
    {
      if (deviceIDs == null || devicesPerWorker < 1)
      {
        return null;
      }

      int[] ret = new int[devicesPerWorker];
      for (int j = 0; j < devicesPerWorker; j++)
      {
        ret[j] = deviceIDs[(workerIndex * devicesPerWorker + j) % deviceIDs.Length];
      }
      return ret;
    }


    /// <summary>
    /// Determines the number of devices per worker when spreading workers over a flat pool of
    /// device IDs: the device count of the engines which will be assigned a slice of the pool,
    /// which must all match. Pooled evaluators and engines without an evaluator are not assigned
    /// a slice and are ignored. Returns 0 if no engine is assigned a slice.
    ///
    /// A warning is issued if the pool is smaller than numConcurrent times the devices per worker
    /// (the device IDs are then reused cyclically).
    /// </summary>
    /// <param name="engineDefs">definitions of the engines to be spread over the pool</param>
    /// <param name="deviceIDs">the pool of device IDs</param>
    /// <param name="numConcurrent">number of concurrent workers</param>
    /// <param name="warn">called with the text of any warning</param>
    /// <returns></returns>
    public static int DevicesPerWorkerForPool(IEnumerable<GameEngineDef> engineDefs, int[] deviceIDs,
                                              int numConcurrent, Action<string> warn)
    {
      int devicesPerWorker = 0;

      foreach (GameEngineDef engineDef in engineDefs)
      {
        NNEvaluatorDef evaluatorDef = engineDef?.GetEvaluatorDef();
        if (evaluatorDef == null || evaluatorDef.DeviceCombo == NNEvaluatorDeviceComboType.Pooled)
        {
          continue; // no evaluator (e.g. an external non-NN engine), or Pooled (shared) -> not sliced
        }

        int numDevices = evaluatorDef.NumDevices;
        if (numDevices < 1)
        {
          throw new Exception($"Engine {engineDef.ID} specification must reference at least one device.");
        }

        if (devicesPerWorker == 0)
        {
          devicesPerWorker = numDevices;
        }
        else if (numDevices != devicesPerWorker)
        {
          throw new Exception($"All engines assigned GPUs from DeviceIDs must reference the same number "
                            + $"of devices (found both {devicesPerWorker} and {numDevices}).");
        }
      }

      if (devicesPerWorker > 0 && deviceIDs.Length < numConcurrent * devicesPerWorker)
      {
        warn?.Invoke($"WARNING: DeviceIDs ({deviceIDs.Length}) fewer than NumConcurrent ({numConcurrent}) "
                   + $"* devices-per-engine ({devicesPerWorker}) = {numConcurrent * devicesPerWorker}; "
                   + $"device IDs will be reused cyclically so concurrent workers share GPU(s) "
                   + $"(deliberate oversubscription; per-engine timing comparability may be reduced).");
      }

      return devicesPerWorker;
    }

    #endregion
  }
}
