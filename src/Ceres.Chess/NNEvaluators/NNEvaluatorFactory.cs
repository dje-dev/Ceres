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
using System.IO;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;

using Ceres.Chess.NNEvaluators.Defs;
using Chess.Ceres.NNEvaluators;
using Ceres.Chess.NNFiles;
using Chess.Ceres.NNEvaluators.TensorRT;
using Ceres.Chess.UserSettings;
using Ceres.Chess.NNEvaluators.CUDA;
using Ceres.Chess.LC0.NNFiles;
using Ceres.Chess.LC0.WeightsProtobuf;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.NNEvaluators.Ceres;
using Ceres.Chess.NNBackends.ONNXRuntime;

#endregion

namespace Ceres.Chess.NNEvaluators
{
    /// <summary>
    /// Static factory methods facilitating construction of NNEvaluators,
    /// including building an NNEvaluator from an NNEvaluatorDef.
    /// </summary>
    public static class NNEvaluatorFactory
  {
    /// <summary>
    /// Delegate type which constructs evaluator from specified definition.
    /// </summary>
    public delegate NNEvaluator CustomDelegate(string netID, int gpuID, NNEvaluator referenceEvaluator, object options);

    /// <summary>
    /// Custom factory method installable at runtime (COMBO_PHASED).
    /// </summary>
    public static CustomDelegate ComboPhasedFactory;

    /// <summary>
    /// Custom factory method installable at runtime (CUSTOM1).
    /// </summary>
    public static CustomDelegate Custom1Factory;

    /// <summary>
    /// Optional options object for use by custom factory method (CUSTOM1). 
    /// </summary>
    public static object Custom1Options; 

    /// <summary>
    /// Custom factory method installable at runtime (CUSTOM2).
    /// </summary>
    public static CustomDelegate Custom2Factory;

    /// <summary>
    /// Optional options object for use by custom factory method (CUSTOM2). 
    /// </summary>
    public static object Custom2Options;


    static Dictionary<object, (NNEvaluatorDef, NNEvaluator)> persistentEvaluators = new();

    internal static void DeletePersistent(NNEvaluator evaluator)
    {
      Debug.Assert(evaluator.IsPersistent);
      persistentEvaluators.Remove(evaluator.PersistentID);
    }


    /// <summary>
    /// Constructs an evaluator based on specified definition,
    /// optionally setting an associated (already initialized) 
    /// reference evaluator which shares the same weights.
    /// </summary>
    /// <param name="def"></param>
    /// <param name="referenceEvaluator"></param>
    /// <returns></returns>
    public static NNEvaluator BuildEvaluator(NNEvaluatorDef def, NNEvaluator referenceEvaluator = null)
    {
      string shortID = null;
      Dictionary<string, string> options = null;
      if (def.OptionsString != null)
      {
        options = new();

        // Parse key value pairs from options string (one or more key=value with semicolon separator).
        foreach (string option in def.OptionsString.Split(';'))
        {
          string[] parts = option.Split('=');
          if (parts.Length == 2)
          {
            if (parts[0] == "ID")
            {
              shortID = parts[1];
            }
            options[parts[0]] = parts[1];
          }
          else if (parts.Length == 1)
          {
            options[parts[0]] = null;
          }
          else
          {
            throw new Exception("Invalid option format: " + option);
          }
        }
      }

      NNEvaluator evaluator = BuildEvaluatorCore(def, referenceEvaluator, options);


      if (shortID != null)
      {
        evaluator.ShortID = shortID;
      }

      if (options != null)
      {
        // Currently on a small number of options are supported.  
        if (options.TryGetValue("ValueTemp", out string valueTemp))
        {
          if (options.Count > 1)
          {
            throw new Exception("Implementation limitation: no other options supported in conjunction with Temperature.");
          }
          evaluator = new NNEvaluatorRemapped(evaluator, valueTemp);
        }
        else if (options.TryGetValue("ZeroHistory", out string zeroHistory))
        {
          if (options.Count > 1)
          {
            throw new Exception("Implementation limitation: no other options supported in conjunction with ZeroHistory.");
          }
          else if (zeroHistory != null)
          {
            throw new NotImplementedException("ZeroHistory option not expected to have associated value.");
          }
          evaluator.ZeroHistoryPlanes = true;
        }

      }

      return evaluator;
    }


    public static NNEvaluator BuildEvaluatorCore(NNEvaluatorDef def, NNEvaluator referenceEvaluator, Dictionary<string, string> options)
    {
      if (def.IsShared)
      {
        lock (persistentEvaluators)
        {
          if (persistentEvaluators.TryGetValue(def.SharedName, out (NNEvaluatorDef persistedEvaluatorDef, NNEvaluator persistedEvaluator) persisted))
          {
            persisted.persistedEvaluator.NumInstanceReferences++;
            return persisted.persistedEvaluator;
          }
          else
          {
            NNEvaluator evaluator = DoBuildEvaluator(def, referenceEvaluator, options);
            evaluator.PersistentID = def.SharedName;
            evaluator.NumInstanceReferences++;
            persistentEvaluators[def.SharedName] = (def, evaluator);
            return evaluator;
          }
        }
      }
      else
      {
        return DoBuildEvaluator(def, referenceEvaluator, options);
      }
    }

    public static void ReleasePersistentEvaluators(string id)
    {
      lock (persistentEvaluators)
      {
        persistentEvaluators.Clear();
      }
    }


    static readonly object onnxFileWriteLock = new();

    static NNEvaluator Singleton(NNEvaluatorNetDef netDef, NNEvaluatorDeviceDef deviceDef, 
                                 NNEvaluator referenceEvaluator, int referenceEvaluatorIndex,
                                 Dictionary<string, string> options)
    {
      NNEvaluator ret = null;

      const bool LOW_PRIORITY = false;
     
      const bool TRT_SHARED = false; // TODO: when is this needed,  when pooled?

      // For ONNX files loaded directly, now way to really know of WDL/MLH present.
      const bool DEFAULT_HAS_WDL = true; 
      const bool DEFAULT_HAS_MLH = false;
      const bool DEFAULT_HAS_UNCERTAINTYV = true;
      const bool DEFAULT_HAS_UNCERTAINTYP = true;
      const bool DEFAULT_HAS_ACTION = false;
      const bool DEFAULT_HAS_STATE = false;

      const int DEFAULT_MAX_BATCH_SIZE = 1024;
      const int TRT_MAX_BATCH_SIZE = 204;// 204; // See note in ONNXExecutor, possibly configuring profile to include large batches hinders performance.
      const bool ONNX_SCALE_50_MOVE_COUNTER = false; // BT2 already inserts own node to adjust

      switch (netDef.Type)
      {
        case NNEvaluatorType.ONNXViaTRT:
        case NNEvaluatorType.ONNXViaORT:
          bool viaTRT = netDef.Type == NNEvaluatorType.ONNXViaTRT
             || (deviceDef.OverrideEngineType != null && deviceDef.OverrideEngineType.StartsWith("TensorRT16"));
          int maxONNXBatchSize = viaTRT ? TRT_MAX_BATCH_SIZE : DEFAULT_MAX_BATCH_SIZE;
          maxONNXBatchSize = Math.Min(maxONNXBatchSize, deviceDef.MaxBatchSize ?? DEFAULT_MAX_BATCH_SIZE);
          string fullFN = Path.Combine(CeresUserSettingsManager.Settings.DirLC0Networks, netDef.NetworkID) + ".onnx";
          //          NNEvaluatorPrecision precision = netDef.NetworkID.EndsWith(".16") ? NNEvaluatorPrecision.FP16 : NNEvaluatorPrecision.FP32;
          ret = new NNEvaluatorONNX(netDef.ShortID, fullFN, null, deviceDef.Type, deviceDef.DeviceIndex, useTRT: viaTRT,
                                            ONNXNetExecutor.NetTypeEnum.LC0, maxONNXBatchSize,
                                            netDef.Precision, DEFAULT_HAS_WDL, DEFAULT_HAS_MLH,
                                            DEFAULT_HAS_UNCERTAINTYV, DEFAULT_HAS_UNCERTAINTYP, DEFAULT_HAS_ACTION,
                                            null, null, null, null, false, ONNX_SCALE_50_MOVE_COUNTER, false, hasState: DEFAULT_HAS_STATE);
          break;

        case NNEvaluatorType.TRT:
          string fullFNTRT = Path.Combine(CeresUserSettingsManager.Settings.DirLC0Networks, netDef.NetworkID) + ".onnx";
          ret = new NNEvaluatorEngineTensorRT(netDef.NetworkID, fullFNTRT, DEFAULT_HAS_WDL, DEFAULT_HAS_MLH, DEFAULT_HAS_UNCERTAINTYV, deviceDef.DeviceIndex,
                                            NNEvaluatorEngineTensorRTConfig.NetTypeEnum.LC0,
                                            deviceDef.MaxBatchSize ?? DEFAULT_MAX_BATCH_SIZE, netDef.Precision,
                                            NNEvaluatorEngineTensorRTConfig.TRTPriorityLevel.Medium, null, false, TRT_SHARED);
          break;

        case NNEvaluatorType.RandomWide:
          ret = new NNEvaluatorRandom(NNEvaluatorRandom.RandomType.WidePolicy, true);
          break;

        case NNEvaluatorType.RandomNarrow:
          ret = new NNEvaluatorRandom(NNEvaluatorRandom.RandomType.NarrowPolicy, true);
          break;

        case NNEvaluatorType.LC0ViaONNXViaORT:
          throw new NotImplementedException();

        case NNEvaluatorType.LC0ViaONNXViaTRT:
          throw new NotImplementedException();

        case NNEvaluatorType.Ceres:
          string[] CERES_ENGINE_TYPES = { "CUDA", "CUDA16", "TENSORRT", "TENSORRT16" };
          if (deviceDef.OverrideEngineType != null && !CERES_ENGINE_TYPES.Contains(deviceDef.OverrideEngineType.ToUpper()))
          {
            throw new Exception($"Ceres engine type not specified or invalid: {deviceDef.OverrideEngineType}." 
              + System.Environment.NewLine + "Valid types: " + string.Join(", ", CERES_ENGINE_TYPES));
          }

          // Temporary hack, Ceres nets requires positions to be retained.
          // TODO: Remove this, or make it an instance variable not global static.
          //       Leaving this globally enabled may impact performance of all evaluators.
          EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true;

          // TODO: Derive these values from NNEvaluatorOptions in the definition object
          const bool ENABLE_PROFILING = false;
          const bool USE_HISTORY = true;
          const bool HAS_UNCERTAINTY_V = true;
          const bool HAS_UNCERTAINTY_P = true;

          // Default is CUDA 16 bit execution, but look for override.
          bool useTensorRT = deviceDef.OverrideEngineType != null && deviceDef.OverrideEngineType.ToUpper().StartsWith("TENSORRT");
          bool useFP16 = !(deviceDef.OverrideEngineType != null && !deviceDef.OverrideEngineType.ToUpper().Contains("16"));

          string onnxFileName = null;

          string shortID = options != null && options.TryGetValue("ID", out string id) ? id : netDef.NetworkID;
          string netFileName = onnxFileName ?? netDef.NetworkID;
          if (!netFileName.ToUpper().EndsWith("ONNX"))
          {
            netFileName += ".onnx";
          }
          if (!File.Exists(netFileName))
          {
            netFileName = Path.Combine(CeresUserSettingsManager.Settings.DirCeresNetworks, netFileName);
          }
          if (!File.Exists(netFileName))
          {
            throw new Exception($"Ceres net {netFileName} not found. Use valid full path or set source directory using DirCeresNetworks in Ceres.json");
          }

          bool testMode = options != null && options.Keys.Contains("TEST");
          NNEvaluatorOptionsCeres optionsCeres = new NNEvaluatorOptionsCeres()
          {
            QNegativeBlunders = 0.02f,
            QPositiveBlunders = 0.02f,
          };

          NNEvaluatorONNX onnxEngine = new(shortID, netFileName, null, 
                                                NNDeviceType.GPU, deviceDef.DeviceIndex, useTensorRT,
                                                ONNXNetExecutor.NetTypeEnum.TPG,
                                                useTensorRT ? TRT_MAX_BATCH_SIZE : DEFAULT_MAX_BATCH_SIZE,
                                                useFP16 ? NNEvaluatorPrecision.FP16 : NNEvaluatorPrecision.FP32,
                                                true, true, HAS_UNCERTAINTY_V, HAS_UNCERTAINTY_P, optionsCeres.UseAction, 
                                                "policy", "value", "mlh", "unc", true,
                                                ENABLE_PROFILING, false, USE_HISTORY, optionsCeres,
                                                true, optionsCeres.UsePriorState);

        EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true; // ** TODO: remove/rework
        onnxEngine.ConverterToFlatFromTPG = (options, o, f1) => TPGConvertersToFlat.ConvertToFlatTPGFromTPG(options, o, f1);
        onnxEngine.ConverterToFlat = (options, o, history, squares, legalMoveIndices) => TPGConvertersToFlat.ConvertToFlatTPG(options, o, history, squares, legalMoveIndices);

        return onnxEngine;

        case NNEvaluatorType.LC0:
          INNWeightsFileInfo net = NNWeightsFiles.LookupNetworkFile(netDef.NetworkID);
          if (CeresUserSettingsManager.Settings.UseLegacyLC0Evaluator)
          {
            ret = new NNEvaluatorLC0(net, deviceDef.DeviceIndex, netDef.Precision);
          }
          else if ((net as NNWeightsFileLC0).Format == NNWeightsFileLC0.FormatType.EmbeddedONNX)
          {
            NNWeightsFileLC0 netDefONNX = net as NNWeightsFileLC0;

            LC0ProtobufNet pbn = LC0ProtobufNet.LoadedNet(net.FileName);
            Debug.Assert(pbn.Net.Format.NetworkFormat.Network == Pblczero.NetworkFormat.NetworkStructure.NetworkOnnx);

            // Extract the ONNX to another file.
            string tempFN = net.FileName.Replace(".pb", "").Replace(".PB", "") + ".onnx";
            if (!File.Exists(tempFN))
            {
              lock (onnxFileWriteLock)
              {
                // string tempFN = Path.GetTempFileName() + ".onnx";
                Console.WriteLine($"The ONNX protobuf from the LC0 PB file is extracted to {tempFN}");
                File.WriteAllBytes(tempFN, pbn.Net.OnnxModel.Model);
              }
            }

            bool useTRT = tempFN.ToUpper().Contains(".TRT"); // TODO: TEMPORARY HACK - way to request using TRT
            if (useTRT)
            {
              return new NNEvaluatorEngineTensorRT(netDef.NetworkID, tempFN, net.IsWDL, net.HasMovesLeft, 
                                                   net.HasUncertaintyV, deviceDef.DeviceIndex,
                                                   NNEvaluatorEngineTensorRTConfig.NetTypeEnum.LC0,
                                                   1024,netDef.Precision, //netDef.Precision,
                                                   NNEvaluatorEngineTensorRTConfig.TRTPriorityLevel.Medium, null, false, TRT_SHARED);
            }
            else
            {
              // TODO: consider if we could/should delete the temporary file when
              //       it has been consumed by the ONNX engine constructor.
              // TODO: consider possibility of other precisions than FP32
              const bool USE_TRT = false;

              //               NNEvaluatorPrecision precision = netDef.NetworkID.Contains(".16") ? NNEvaluatorPrecision.FP16 : NNEvaluatorPrecision.FP32;
              return new NNEvaluatorONNX(netDef.NetworkID, tempFN, null, deviceDef.Type, deviceDef.DeviceIndex, USE_TRT,
                                               ONNXNetExecutor.NetTypeEnum.LC0, 1024,
                                               netDef.Precision, netDefONNX.IsWDL, netDefONNX.HasMovesLeft, 
                                               netDefONNX.HasUncertaintyV, netDefONNX.HasUncertaintyP, false,
                                               pbn.Net.OnnxModel.OutputValue, pbn.Net.OnnxModel.OutputWdl,
                                               pbn.Net.OnnxModel.OutputPolicy, pbn.Net.OnnxModel.OutputMlh, false, ONNX_SCALE_50_MOVE_COUNTER,
                                               false);
            }
          }
          else
          {
            NNEvaluatorCUDA referenceEvaluatorCast = null;

            if (referenceEvaluator != null)
            {
              if (referenceEvaluator is NNEvaluatorCUDA)
              {
                referenceEvaluatorCast = referenceEvaluator as NNEvaluatorCUDA;
              }
              else if (referenceEvaluator is NNEvaluatorSplit)
              {
                referenceEvaluatorCast = ((NNEvaluatorSplit)referenceEvaluator).Evaluators[referenceEvaluatorIndex] as NNEvaluatorCUDA;
              }
            }

            // Determine max batch size.
            int maxSearchBatchSize = CeresUserSettingsManager.Settings.MaxBatchSize;
            int maxBatchSize = maxSearchBatchSize;
            if (deviceDef.MaxBatchSize.HasValue)
            {
              maxBatchSize = Math.Min(deviceDef.MaxBatchSize.Value, maxSearchBatchSize);
            }

            bool enableCUDAGraphs = CeresUserSettingsManager.Settings.EnableCUDAGraphs;
            NNEvaluatorCUDA evaluator = new(net as NNWeightsFileLC0, deviceDef.DeviceIndex, maxBatchSize,
                                             false, netDef.Precision, false, enableCUDAGraphs, 1, referenceEvaluatorCast);

            // Possibly specialize as NNEvaluatorSubBatchedWithTarget if device batch size is set.
            bool useTargetedEvaluator = (deviceDef.PredefinedOptimalBatchPartitions != null
                                      || (deviceDef.MaxBatchSize.HasValue && deviceDef.MaxBatchSize.Value < maxSearchBatchSize));
            if (useTargetedEvaluator)
            {
              return new NNEvaluatorSubBatchedWithTarget(evaluator, 
                                                         deviceDef.MaxBatchSize.Value, 
                                                         deviceDef.OptimalBatchSize.Value, 
                                                         deviceDef.PredefinedOptimalBatchPartitions);
#if NOT
          NNEvaluator copySinglegon = Singleton(def.Nets[0].Net, def.Devices[0].Device, referenceEvaluator);
          NNEvaluatorCompare compare = new NNEvaluatorCompare(subbatched, copySinglegon);
          compare.RunParallel = false;
          return compare;
#endif
            }
            else
            {
              return evaluator;
            }

          }

          break;

        case NNEvaluatorType.ComboPhased:
          if (ComboPhasedFactory == null)
          {
            throw new Exception("NNEvaluatorFactory.ComboPhasedFactory static variable must be initialized.");
          }
          ret = ComboPhasedFactory(netDef.NetworkID, deviceDef.DeviceIndex, referenceEvaluator, Custom1Options);
          break;

        case NNEvaluatorType.Custom1:
          if (Custom1Factory == null)
          {
            throw new Exception("NNEvaluatorFactory.Custom1Factory static variable must be initialized.");
          }
          ret = Custom1Factory(netDef.NetworkID, deviceDef.DeviceIndex, referenceEvaluator, Custom1Options);
          break;

        case NNEvaluatorType.Custom2:
          if (Custom2Factory == null)
          {
            throw new Exception("NNEvaluatorFactory.Custom2Factory static variable must be initialized.");
          }
          ret = Custom2Factory(netDef.NetworkID, deviceDef.DeviceIndex, referenceEvaluator, Custom2Options);
          break;

        default:
          throw new Exception($"Requested neural network evaluator type not supported: {netDef.Type}");
      }

      return  ret;
    }


    static NNEvaluator BuildDeviceCombo(NNEvaluatorDef def, NNEvaluator referenceEvaluator, Dictionary<string, string> options)
    {
      Debug.Assert(def.Nets.Length == 1);

      // Build underlying device evaluators in parallel
      NNEvaluator[] evaluators = new NNEvaluator[def.Devices.Length];
      float[] fractions = new float[def.Devices.Length];

      try
      {
        // Option to disable if problems are seen
        const bool PARALLEL_INIT_ENABLED = true;
  
        Parallel.For(0, def.Devices.Length, new ParallelOptions() { MaxDegreeOfParallelism = PARALLEL_INIT_ENABLED ? int.MaxValue : 1 }, 
          delegate (int i)
          {
            if (def.Nets.Length == 1)
            {
              evaluators[i] = Singleton(def.Nets[0].Net, def.Devices[i].Device, referenceEvaluator, i, options);
            }
            else
            {
             evaluators[i] = BuildNetCombo(def, referenceEvaluator, options);
            }

          fractions[i] = def.Devices[i].Fraction;
          });
      }
      catch (Exception exc)
      {
        throw new Exception($"Exception in initialization of NNEvaluator: {def}" + exc);
      }

      // Combine together devices
      return def.DeviceCombo switch
      {
        NNEvaluatorDeviceComboType.Split      => new NNEvaluatorSplit(evaluators, fractions, def.MinSplitNumPositions),
        NNEvaluatorDeviceComboType.RoundRobin => new NNEvaluatorRoundRobin(evaluators),
        NNEvaluatorDeviceComboType.Pooled     => new NNEvaluatorPooled(evaluators),
        NNEvaluatorDeviceComboType.Compare    => new NNEvaluatorCompare(evaluators),
        _ => throw new NotImplementedException()
      };
    }


    static NNEvaluator BuildNetCombo(NNEvaluatorDef def, NNEvaluator referenceEvaluator, Dictionary<string, string> options)
    {
      Debug.Assert(def.Devices.Length == 1);

      // Build underlying device evaluators in parallel
      NNEvaluator[] evaluators = new NNEvaluator[def.Nets.Length];
      float[] weightsValue = new float[def.Nets.Length];
      float[] weightsValue2 = new float[def.Nets.Length];
      float[] weightsPolicy = new float[def.Nets.Length];
      float[] weightsM = new float[def.Nets.Length];
      float[] weightsU = new float[def.Nets.Length];
      float[] weightsUPolicy = new float[def.Nets.Length];
      Parallel.For(0, def.Nets.Length, delegate (int i)
      {
        evaluators[i] = Singleton(def.Nets[i].Net, def.Devices[0].Device, referenceEvaluator, i, options);
        weightsValue[i] = def.Nets[i].WeightValue;
        weightsValue2[i] = def.Nets[i].WeightValue2;
        weightsPolicy[i] = def.Nets[i].WeightPolicy;
        weightsM[i] = def.Nets[i].WeightM;
        weightsU[i] = def.Nets[i].WeightU;
        weightsUPolicy[i] = def.Nets[i].WeightUPolicy;
      });

      return def.NetCombo switch
      {
        NNEvaluatorNetComboType.WtdAverage => new NNEvaluatorLinearCombo(evaluators, weightsValue, weightsValue2, 
                                                                         weightsPolicy, weightsM, weightsU, weightsUPolicy, null),
        NNEvaluatorNetComboType.Compare    => new NNEvaluatorCompare(evaluators),
        _ => throw new NotImplementedException()
      };
    }


    static NNEvaluator DoBuildEvaluator(NNEvaluatorDef def, NNEvaluator referenceEvaluator, Dictionary<string, string> options)
    {
      if (def.DeviceCombo == NNEvaluatorDeviceComboType.Single && def.DeviceIndices.Length > 1)
      {
        throw new Exception("DeviceComboType.Single is not expected when number of DeviceIndices is greater than 1.");
      }

      if (def.NetCombo == NNEvaluatorNetComboType.Dynamic)
      {
        throw new NotImplementedException("Dynamic not yet implemented");
      }

   
      //        if (Params.EstimatePerformanceCharacteristics) ret.CalcStatistics(true);

      // TODO: restore implementation, test more to see if faster or memory efficient
      const bool LC0_SERVER_ENABLE_MULTI_GPU = false; // seems somewhat slower
      if (LC0_SERVER_ENABLE_MULTI_GPU
            && def.NetCombo == NNEvaluatorNetComboType.Single
            && def.Devices.Length > 1
            && def.Devices[0].Device.Type == NNDeviceType.GPU 
            && def.Nets[0].Net.Type == NNEvaluatorType.LC0)
      {
        int[] deviceIndices = new int[def.Devices.Length];
        for (int i=0; i < def.Devices.Length; i++)
        {
          throw new NotImplementedException();
        }
      }

      if (def.DeviceCombo != NNEvaluatorDeviceComboType.Single)
      {
        return BuildDeviceCombo(def, referenceEvaluator, options);
      }
      else if (def.NetCombo != NNEvaluatorNetComboType.Single)
      {
        return BuildNetCombo(def, referenceEvaluator, options);
      }
      else
      {
        return Singleton(def.Nets[0].Net, def.Devices[0].Device, referenceEvaluator, 0, options);
      }
    }
 

  }
}
