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

using Ceres.Chess.NNEvaluators.Defs;
using Chess.Ceres.NNEvaluators;
using Ceres.Chess.NNFiles;
using System.Collections.Generic;
using Chess.Ceres.NNEvaluators.TensorRT;
using Ceres.Chess.UserSettings;
using Ceres.Chess.NNEvaluators.CUDA;
using Ceres.Chess.LC0.NNFiles;
using Ceres.Chess.LC0NetInference;
using Ceres.Chess.LC0.WeightsProtobuf;
using Ceres.Chess.LC0.Batches;

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
    public delegate NNEvaluator CustomDelegate(string netID, int gpuID, NNEvaluator referenceEvaluator);

    /// <summary>
    /// Custom factory method installable at runtime (COMBO_PHASED).
    /// </summary>
    public static CustomDelegate ComboPhasedFactory;

    /// <summary>
    /// Custom factory method installable at runtime (CUSTOM1).
    /// </summary>
    public static CustomDelegate Custom1Factory;

    /// <summary>
    /// Custom factory method installable at runtime (CUSTOM2).
    /// </summary>
    public static CustomDelegate Custom2Factory;


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
            NNEvaluator evaluator = DoBuildEvaluator(def, referenceEvaluator);
            evaluator.PersistentID = def.SharedName;
            evaluator.NumInstanceReferences++;
            persistentEvaluators[def.SharedName] = (def, evaluator);
            return evaluator;
          }
        }
      }
      else
      {
        return DoBuildEvaluator(def, referenceEvaluator);
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
                                 NNEvaluator referenceEvaluator, int referenceEvaluatorIndex = 0)
    {
      NNEvaluator ret = null;

      const bool LOW_PRIORITY = false;
     
      const bool TRT_SHARED = false; // TODO: when is this needed,  when pooled?

      // For ONNX files loaded directly, now way to really know of WDL/MLH present.
      const bool DEFAULT_HAS_WDL = true; 
      const bool DEFAULT_HAS_MLH = true; 

      switch (netDef.Type)
      {
        case NNEvaluatorType.ONNXViaTRT:
          ret = new NNEvaluatorEngineTensorRT(netDef.NetworkID, netDef.NetworkID, DEFAULT_HAS_WDL, DEFAULT_HAS_MLH, deviceDef.DeviceIndex,
                                              NNEvaluatorEngineTensorRTConfig.NetTypeEnum.LC0,
                                              1024, netDef.Precision,
                                              NNEvaluatorEngineTensorRTConfig.TRTPriorityLevel.Medium, null, false, TRT_SHARED);
          break;

        case NNEvaluatorType.ONNXViaORT:
          // TODO: consider possibility of other precisions than FP32
          ret = new NNEvaluatorEngineONNX(netDef.NetworkID, netDef.NetworkID, deviceDef.DeviceIndex,
                                          ONNXRuntimeExecutor.NetTypeEnum.LC0, 1024,
                                          NNEvaluatorPrecision.FP32, DEFAULT_HAS_WDL, DEFAULT_HAS_MLH,
                                          null, null, null, null);
          break;

        case NNEvaluatorType.RandomWide:
          ret = new NNEvaluatorRandom(NNEvaluatorRandom.RandomType.WidePolicy, true);
          break;

        case NNEvaluatorType.RandomNarrow:
          ret = new NNEvaluatorRandom(NNEvaluatorRandom.RandomType.NarrowPolicy, true);
          break;

        case NNEvaluatorType.LC0Library:
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
                File.WriteAllBytes(tempFN, pbn.Net.OnnxModel.Model);
              }
            }

            bool useTRT = !tempFN.ToUpper().Contains(".ORT"); // TODO: TEMPORARY HACK - way to request using ORT
            if (useTRT)
            {
              return new NNEvaluatorEngineTensorRT(netDef.NetworkID, tempFN, net.IsWDL, net.HasMovesLeft, deviceDef.DeviceIndex,
                                                   NNEvaluatorEngineTensorRTConfig.NetTypeEnum.LC0,
                                                   1024, netDef.Precision,
                                                   NNEvaluatorEngineTensorRTConfig.TRTPriorityLevel.Medium, null, false, TRT_SHARED);
            }
            else
            {
              // TODO: consider if we could/should delete the temporary file when
              //       it has been consumed by the ONNX engine constructor.
              // TODO: consider possibility of other precisions than FP32
              return new NNEvaluatorEngineONNX(netDef.NetworkID, tempFN, deviceDef.DeviceIndex, 
                                               ONNXRuntimeExecutor.NetTypeEnum.LC0, 1024,
                                               NNEvaluatorPrecision.FP32, netDefONNX.IsWDL, netDefONNX.HasMovesLeft,
                                               pbn.Net.OnnxModel.OutputValue, pbn.Net.OnnxModel.OutputWdl,
                                               pbn.Net.OnnxModel.OutputPolicy, pbn.Net.OnnxModel.OutputMlh);
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


            int maxBatchSize = CeresUserSettingsManager.Settings.MaxBatchSize;
            bool enableCUDAGraphs = CeresUserSettingsManager.Settings.EnableCUDAGraphs;
            return new NNEvaluatorCUDA(net as NNWeightsFileLC0, deviceDef.DeviceIndex, maxBatchSize, 
                                       false, netDef.Precision, false, enableCUDAGraphs, 1, referenceEvaluatorCast);
          }

          break;

        case NNEvaluatorType.ComboPhased:
          if (ComboPhasedFactory == null)
          {
            throw new Exception("NNEvaluatorFactory.ComboPhasedFactory static variable must be initialized.");
          }
          ret = ComboPhasedFactory(netDef.NetworkID, deviceDef.DeviceIndex, referenceEvaluator);
          break;

        case NNEvaluatorType.Custom1:
          if (Custom1Factory == null)
          {
            throw new Exception("NNEvaluatorFactory.Custom1Factory static variable must be initialized.");
          }
          INNWeightsFileInfo netInfoCustom1 = NNWeightsFiles.LookupNetworkFile(netDef.NetworkID);
          ret = Custom1Factory(netInfoCustom1.NetworkID, deviceDef.DeviceIndex, referenceEvaluator);
          break;

        case NNEvaluatorType.Custom2:
          INNWeightsFileInfo netInfoCustom2 = NNWeightsFiles.LookupNetworkFile(netDef.NetworkID);
          if (Custom2Factory == null)
          {
            throw new Exception("NNEvaluatorFactory.Custom2Factory static variable must be initialized.");
          }
          ret = Custom2Factory(netInfoCustom2.NetworkID, deviceDef.DeviceIndex, referenceEvaluator);
          break;

        default:
          throw new Exception($"Requested neural network evaluator type not supported: {netDef.Type}");
      }

      return  ret;
    }


    static NNEvaluator BuildDeviceCombo(NNEvaluatorDef def, NNEvaluator referenceEvaluator)
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
              evaluators[i] = Singleton(def.Nets[0].Net, def.Devices[i].Device, referenceEvaluator, i);
            }
            else
            {
              throw new NotImplementedException();
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


    static NNEvaluator BuildNetCombo(NNEvaluatorDef def, NNEvaluator referenceEvaluator)
    {
      Debug.Assert(def.Devices.Length == 1);

      // Build underlying device evaluators in parallel
      NNEvaluator[] evaluators = new NNEvaluator[def.Nets.Length];
      float[] weightsValue = new float[def.Nets.Length];
      float[] weightsPolicy = new float[def.Nets.Length];
      float[] weightsM = new float[def.Nets.Length];
      Parallel.For(0, def.Nets.Length, delegate (int i)
      {
        evaluators[i] = Singleton(def.Nets[i].Net, def.Devices[0].Device, referenceEvaluator);
        weightsValue[i] = def.Nets[i].WeightValue;
        weightsPolicy[i] = def.Nets[i].WeightPolicy;
        weightsM[i] = def.Nets[i].WeightM;
      });

      return def.NetCombo switch
      {
        NNEvaluatorNetComboType.WtdAverage => new NNEvaluatorLinearCombo(evaluators, weightsValue, weightsPolicy, weightsM, null, null),
        NNEvaluatorNetComboType.Compare    => new NNEvaluatorCompare(evaluators),
        _ => throw new NotImplementedException()
      };
    }

    static NNEvaluator DoBuildEvaluator(NNEvaluatorDef def, NNEvaluator referenceEvaluator)
    {
      if (def.DeviceCombo == NNEvaluatorDeviceComboType.Single && def.DeviceIndices.Length > 1)
      {
        throw new Exception("DeviceComboType.Single is not expected when number of DeviceIndices is greater than 1.");
      }

      if (def.NetCombo == NNEvaluatorNetComboType.Dynamic)
      {
        throw new NotImplementedException("Dynamic not yet implemented");
      }

      if (def.NetCombo != NNEvaluatorNetComboType.Single && def.DeviceCombo != NNEvaluatorDeviceComboType.Single)
      {
        throw new NotImplementedException("Currently either NetComboType or DeviceComboType must be Single.");
      }

      //        if (Params.EstimatePerformanceCharacteristics) ret.CalcStatistics(true);

      // TODO: restore implementation, test more to see if faster or memory efficient
      const bool LC0_SERVER_ENABLE_MULTI_GPU = false; // seems somewhat slower
      if (LC0_SERVER_ENABLE_MULTI_GPU
            && def.NetCombo == NNEvaluatorNetComboType.Single
            && def.Devices.Length > 1
            && def.Devices[0].Device.Type == NNDeviceType.GPU 
            && def.Nets[0].Net.Type == NNEvaluatorType.LC0Library)
      {
        int[] deviceIndices = new int[def.Devices.Length];
        for (int i=0; i < def.Devices.Length; i++)
        {
          throw new NotImplementedException();
        }
      }

      if (def.DeviceCombo != NNEvaluatorDeviceComboType.Single)
      {
        return BuildDeviceCombo(def, referenceEvaluator);
      }
      else if (def.NetCombo != NNEvaluatorNetComboType.Single)
      {
        return BuildNetCombo(def, referenceEvaluator);
      }
      else
      {
        return Singleton(def.Nets[0].Net, def.Devices[0].Device, referenceEvaluator);
      }
    }
 

  }
}
