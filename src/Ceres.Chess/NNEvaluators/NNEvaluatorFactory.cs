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
    /// Custom factory method installable at runtime (CUSTOM1).
    /// </summary>
    public static Func<string, int, NNEvaluator> Custom1Factory;

    static Dictionary<object, (NNEvaluatorDef, NNEvaluator)> persistentEvaluators = new();

    internal static void DeletePersistent(NNEvaluator evaluator)
    {
      Debug.Assert(evaluator.IsPersistent);
      persistentEvaluators.Remove(evaluator.PersistentID);
    }

    public static NNEvaluator BuildEvaluator(NNEvaluatorDef def)
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
            NNEvaluator evaluator = DoBuildEvaluator(def);
            evaluator.PersistentID = def.persistentID;
            evaluator.NumInstanceReferences++;
            persistentEvaluators[def.persistentID] = (def, evaluator);
            return evaluator;
          }
        }
      }
      else
      {
        return DoBuildEvaluator(def);
      }
    }

    public static void ReleasePersistentEvaluators(string id)
    {
      lock (persistentEvaluators)
      {
        persistentEvaluators.Clear();
      }
    }

    static NNEvaluator Singleton(NNEvaluatorNetDef netDef, NNEvaluatorDeviceDef deviceDef)
    {
      NNEvaluator ret = null;

      const bool LOW_PRIORITY = false;
      //LC0DownloadedNetDef net = LC0DownloadedNetDef.ByID(netDef.NetworkID);
      INNWeightsFileInfo net = null;

      // TODO: also do this for ONNX
      if( netDef.Type != NNEvaluatorType.ONNX) net = NNWeightsFiles.LookupNetworkFile(netDef.NetworkID);

      if (netDef.Type == NNEvaluatorType.RandomWide)
        ret = new NNEvaluatorRandom(NNEvaluatorRandom.RandomType.WidePolicy, true);
      else if (netDef.Type == NNEvaluatorType.RandomNarrow)
        ret = new NNEvaluatorRandom(NNEvaluatorRandom.RandomType.NarrowPolicy, true);
      else if (netDef.Type == NNEvaluatorType.LC0Library)
        ret = new NNEvaluatorLC0(net, deviceDef.DeviceIndex, netDef.Precision);

      else if (netDef.Type == NNEvaluatorType.LC0TensorRT)
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
      }
      else if (netDef.Type == NNEvaluatorType.ONNX)
      {
        // TODO: fill these in properly
        string fn = @$"C:\dev\CeresDev\src\Ceres.TFTrain\{netDef.NetworkID}.onnx";
        bool isWDL = true;
        bool hasMLH = true;
        ret = new NNEvaluatorEngineONNX(netDef.NetworkID, fn, deviceDef.DeviceIndex,
                                        ONNXRuntimeExecutor.NetTypeEnum.LC0, 1024, isWDL, hasMLH);
      }
      else if (netDef.Type == NNEvaluatorType.Custom1)
      {
        if (Custom1Factory == null)
        {
          throw new Exception("NNEvaluatorFactory.Custom1Factory static variable must be initialized.");
        }
        ret = Custom1Factory(net.NetworkID, deviceDef.DeviceIndex);
      }
      else
      {
        throw new Exception($"Requested neural network evaluator type not supported: {netDef.Type}");
      }

      return  ret;
    }


    static NNEvaluator BuildDeviceCombo(NNEvaluatorDef def)
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
            evaluators[i] = Singleton(def.Nets[0].Net, def.Devices[i].Device);
          else
            throw new NotImplementedException();
        //evaluators[i] = BuildNetCombo(def.Nets[0].Net, def.Devices[i].Device);

        fractions[i] = def.Devices[i].Fraction;
        });
      }
      catch (Exception exc)
      {
        throw new Exception($"Exception in initialization of NNEvaluator: {def}" + exc);
      }

      // Combine together devices
      if (def.DeviceCombo == NNEvaluatorDeviceComboType.Split)
        return new NNEvaluatorSplit(evaluators, fractions, def.MinSplitNumPositions);
      else if (def.DeviceCombo == NNEvaluatorDeviceComboType.RoundRobin)
        return new NNEvaluatorRoundRobin(evaluators);
      else if (def.DeviceCombo == NNEvaluatorDeviceComboType.Pooled)
      {
//        const int MULTIBATCH_THRESHOLD_NUM_POSITIONS = 150;// 128;
//        const int MULTIBATCH_THRESHOLD_WAIT_TIME_MS = 10;// 2; 

        return new NNEvaluatorPooled(evaluators);//, MULTIBATCH_THRESHOLD_NUM_POSITIONS, MULTIBATCH_THRESHOLD_WAIT_TIME_MS, def.RetrieveSupplementalLayers);
      }
      else if (def.DeviceCombo == NNEvaluatorDeviceComboType.Compare)
        return new NNEvaluatorCompare(evaluators);
      else
        throw new NotImplementedException();
    }


    static NNEvaluator BuildNetCombo(NNEvaluatorDef def)
    {
      Debug.Assert(def.Devices.Length == 1);

      // Build underlying device evaluators in parallel
      NNEvaluator[] evaluators = new NNEvaluator[def.Nets.Length];
      float[] weightsValue = new float[def.Nets.Length];
      float[] weightsPolicy = new float[def.Nets.Length];
      float[] weightsM = new float[def.Nets.Length];
      Parallel.For(0, def.Nets.Length, delegate (int i)
      {
        evaluators[i] = Singleton(def.Nets[i].Net, def.Devices[0].Device);
        weightsValue[i] = def.Nets[i].WeightValue;
        weightsPolicy[i] = def.Nets[i].WeightPolicy;
        weightsM[i] = def.Nets[i].WeightM;
      });

      if (def.NetCombo == NNEvaluatorNetComboType.WtdAverage)
        return new NNEvaluatorLinearCombo(evaluators, weightsValue, weightsPolicy, weightsM, null, null);
      else if (def.NetCombo == NNEvaluatorNetComboType.Compare)
        return new NNEvaluatorCompare(evaluators);
      else
        throw new NotImplementedException();
    }

     static NNEvaluator DoBuildEvaluator(NNEvaluatorDef def)
    {
      if (def.DeviceCombo == NNEvaluatorDeviceComboType.Single && def.DeviceIndices.Length > 1)
        throw new Exception("DeviceComboType.Single is not expected when number of DeviceIndices is greater than 1.");

      if (def.NetCombo == NNEvaluatorNetComboType.Dynamic)
        throw new NotImplementedException("Dynamic not yet implemented");

      if (def.NetCombo != NNEvaluatorNetComboType.Single && def.DeviceCombo != NNEvaluatorDeviceComboType.Single)
        throw new NotImplementedException("Currently either NetComboType or DeviceComboType must be Single.");

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
        for (int i=0; i < def.Devices.Length;i++)
        {
          throw new NotImplementedException();
          //if (def.Devices[i].Type)
        }

        INNWeightsFileInfo netDef = NNWeightsFiles.LookupNetworkFile(def.Nets[0].Net.NetworkID);
        return new NNEvaluatorLC0(netDef, deviceIndices, def.Nets[0].Net.Precision);
//        return new NNEvaluatorLC0Server(netDef, deviceIndices, def.Nets[0].Net.Precision);
      }

      if (def.DeviceCombo != NNEvaluatorDeviceComboType.Single)
        return BuildDeviceCombo(def);
      else if (def.NetCombo != NNEvaluatorNetComboType.Single)
        return BuildNetCombo(def);
      else
        return Singleton(def.Nets[0].Net, def.Devices[0].Device);

    }
  }
}
