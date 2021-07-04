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
using System.Diagnostics;
using System.Threading.Tasks;

using Ceres.Chess.NNEvaluators.Defs;
using Chess.Ceres.NNEvaluators;
using Ceres.Chess.NNFiles;
using System.Collections.Generic;
using Chess.Ceres.NNEvaluators.TensorRT;
using Ceres.Chess.LC0NetInference;
using Ceres.Chess.UserSettings;
using Ceres.Chess.NNEvaluators.CUDA;
using Ceres.Chess.LC0.NNFiles;

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
      if (def.IsPersistent)
      {
        lock (persistentEvaluators)
        {
          if (persistentEvaluators.TryGetValue(def.persistentID, out (NNEvaluatorDef persistedEvaluatorDef, NNEvaluator persistedEvaluator) persisted))
          {
            persisted.persistedEvaluator.NumInstanceReferences++;
            return persisted.persistedEvaluator;
          }
          else
          {
            NNEvaluator evaluator = DoBuildEvaluator(def, referenceEvaluator);
            evaluator.PersistentID = def.persistentID;
            evaluator.NumInstanceReferences++;
            persistentEvaluators[def.persistentID] = (def, evaluator);
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

    static NNEvaluator Singleton(NNEvaluatorNetDef netDef, NNEvaluatorDeviceDef deviceDef, NNEvaluator referenceEvaluator)
    {
      NNEvaluator ret = null;

      const bool LOW_PRIORITY = false;
      
      INNWeightsFileInfo net = null;

      // TODO: also do this for ONNX
      if (netDef.Type != NNEvaluatorType.ONNX)
      {
        net = NNWeightsFiles.LookupNetworkFile(netDef.NetworkID);
      }

      switch (netDef.Type)
      {
        case NNEvaluatorType.RandomWide:
          ret = new NNEvaluatorRandom(NNEvaluatorRandom.RandomType.WidePolicy, true);
          break;

        case NNEvaluatorType.RandomNarrow:
          ret = new NNEvaluatorRandom(NNEvaluatorRandom.RandomType.NarrowPolicy, true);
          break;

        case NNEvaluatorType.LC0Library:
          if (CeresUserSettingsManager.Settings.UseLegacyLC0Evaluator)
          {
            ret = new NNEvaluatorLC0(net, deviceDef.DeviceIndex, netDef.Precision);
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
                referenceEvaluatorCast = ((NNEvaluatorSplit)referenceEvaluator).Evaluators[0] as NNEvaluatorCUDA;
              }
            }


            int maxBatchSize = CeresUserSettingsManager.Settings.MaxBatchSize;
            bool enableCUDAGraphs = CeresUserSettingsManager.Settings.EnableCUDAGraphs;
            return new NNEvaluatorCUDA(net as NNWeightsFileLC0, deviceDef.DeviceIndex, maxBatchSize, 
                                       false, netDef.Precision, false, enableCUDAGraphs, 1, referenceEvaluatorCast);
          }

          break;

        case NNEvaluatorType.LC0TensorRT:
          {
            NNEvaluatorEngineTensorRTConfig.TRTPriorityLevel priority = LOW_PRIORITY ? NNEvaluatorEngineTensorRTConfig.TRTPriorityLevel.Medium
                                                                                     : NNEvaluatorEngineTensorRTConfig.TRTPriorityLevel.High;
            const int MAX_TRT_BATCH_SIZE = 1024; // TODO: move this elsewhere
            const bool SHARED = false;
            const bool USE_MULTI = false; // attempt to to workaround apparent bug in TRT where cannot change batch size (probably unusccessful)
            if (USE_MULTI)
            {
              throw new NotImplementedException();
              //          ret = new NNEvaluatorEngineTensorRTMultiBatchSizes(net.NetworkID, net.ONNXFileName, net.IsWDL, net.HasMovesLeft, deviceDef.DeviceIndex,
              //                                                             NNEvaluatorEngineTensorRTConfig.NetTypeEnum.LC0,
              //                                                             MAX_TRT_BATCH_SIZE, netDef.Precision, priority, null, shared: SHARED);
            }
            else
            {
              ret = new NNEvaluatorEngineTensorRT(net.NetworkID, net.ONNXFileName, net.IsWDL, net.HasMovesLeft, deviceDef.DeviceIndex,
                                                  NNEvaluatorEngineTensorRTConfig.NetTypeEnum.LC0,
                                                  MAX_TRT_BATCH_SIZE, netDef.Precision, priority, null, shared: SHARED);
            }

            break;
          }

        case NNEvaluatorType.ONNX:
          {
            // TODO: fill these in properly
            string fn = @$"d:\weights\lczero.org\{netDef.NetworkID}.onnx";
            bool isWDL = true;
            bool hasMLH =  false; // TODO: fix. Eventually put in netDef
            ret = new NNEvaluatorEngineONNX(netDef.NetworkID, fn, deviceDef.DeviceIndex,
                                            ONNXRuntimeExecutor.NetTypeEnum.LC0, 1024, netDef.Precision, isWDL, hasMLH);
            break;
          }

        case NNEvaluatorType.Custom1:
          if (Custom1Factory == null)
          {
            throw new Exception("NNEvaluatorFactory.Custom1Factory static variable must be initialized.");
          }
          ret = Custom1Factory(net.NetworkID, deviceDef.DeviceIndex, referenceEvaluator);
          break;

        case NNEvaluatorType.Custom2:
          if (Custom2Factory == null)
          {
            throw new Exception("NNEvaluatorFactory.Custom2Factory static variable must be initialized.");
          }
          ret = Custom2Factory(net.NetworkID, deviceDef.DeviceIndex, referenceEvaluator);
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
        Parallel.For(0, def.Devices.Length, delegate (int i)
        {
          if (def.Nets.Length == 1)
          {
            evaluators[i] = Singleton(def.Nets[0].Net, def.Devices[i].Device, referenceEvaluator);
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
