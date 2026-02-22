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
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.LC0.NNFiles;
using Ceres.Chess.LC0.WeightsProtobuf;
using Ceres.Chess.NNBackends.ONNXRuntime;
using Ceres.Chess.NNEvaluators.Ceres;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.NNEvaluators.CUDA;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.TensorRT;
using Ceres.Chess.NNFiles;
using Ceres.Chess.UserSettings;
using Chess.Ceres.NNEvaluators;
using Chess.Ceres.NNEvaluators.TensorRT;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Static factory methods facilitating construction of NNEvaluators,
  /// including building an NNEvaluator from an NNEvaluatorDef.
  /// </summary>
  public static class NNEvaluatorFactory
  {
    #region Installable evaluator factories

    /// <summary>
    /// Delegate type which constructs evaluator from specified definition.
    /// </summary>
    public delegate NNEvaluator CustomDelegate(string netID,
                                               int gpuID,
                                               NNEvaluator referenceEvaluator,
                                               object options,
                                               Dictionary<string, string> optionsDict);

    /// <summary>
    /// Custom factory method installable at runtime (COMBO_PHASED).
    /// </summary>
    public static CustomDelegate ComboPhasedFactory;

    /// <summary>
    /// Optional options object for use by combo phased custom factory.
    /// </summary>
    public static NNEvaluatorOptions CustomPhasedOptions;

    /// <summary>
    /// Custom factory method installable at runtime (CUSTOM1).
    /// </summary>
    public static CustomDelegate Custom1Factory;

    /// <summary>
    /// Optional options object for use by custom factory method (CUSTOM1). 
    /// </summary>
    public static NNEvaluatorOptions Custom1Options;

    /// <summary>
    /// Custom factory method installable at runtime (CUSTOM2).
    /// </summary>
    public static CustomDelegate Custom2Factory;

    /// <summary>
    /// Optional options object for use by custom factory method (CUSTOM2). 
    /// </summary>
    public static NNEvaluatorOptions Custom2Options;


    /// <summary>
    /// Delegate type of method that will construct an NNEvaluator which uses the Torchscript evaluator.
    /// </summary>
    /// <param name="fn"></param>
    /// <param name="gpuIndex"></param>
    /// <param name="options"></param>
    /// <param name="useBFloat"></param>
    /// <returns></returns>
    public delegate NNEvaluator BuildTorchscriptEvaluatorDelegate(string fn,
                                                                  int gpuIndex,
                                                                  NNEvaluatorOptionsCeres options,
                                                                  bool useBFloat = false);

    /// <summary>
    /// Public static installable method to construct Torchscript evaluator.
    /// The use of an installed method allows us to avoid adding the large TorchSharp NuGet package to the project.
    /// </summary>
    public static BuildTorchscriptEvaluatorDelegate BuildTorchscriptEvaluator = null;


    #endregion


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
            options[parts[0].ToUpper()] = parts[1];
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
      if (def.Options != null)
      {
        evaluator.Options = def.Options;
      }

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
                                 Dictionary<string, string> optionsDict,
                                 NNEvaluatorOptions optionsEvaluator)
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
      const int TRT_MAX_BATCH_SIZE = 1024; // See note in ONNXExecutor, possibly configuring profile to include large batches hinders performance.
      const bool ONNX_SCALE_50_MOVE_COUNTER = false; // BT2 already inserts own node to adjust

      bool isTensorRTNative = deviceDef.OverrideEngineType?.Contains("TensorRTNative", StringComparison.OrdinalIgnoreCase) == true;

      switch (netDef.Type)
      {
        case NNEvaluatorType.ONNXViaTRT:
        case NNEvaluatorType.ONNXViaORT:

          bool viaTRT = netDef.Type == NNEvaluatorType.ONNXViaTRT
             || (deviceDef.OverrideEngineType != null && deviceDef.OverrideEngineType.StartsWith("TensorRT16"));
          int maxONNXBatchSize = viaTRT ? TRT_MAX_BATCH_SIZE : DEFAULT_MAX_BATCH_SIZE;
          maxONNXBatchSize = Math.Min(maxONNXBatchSize, deviceDef.MaxBatchSize ?? DEFAULT_MAX_BATCH_SIZE);
          bool netIDExistsAsIS = File.Exists(netDef.NetworkID) || File.Exists(netDef.NetworkID + ".onnx");
          string pathLc0Networks = CeresUserSettingsManager.Settings.DirLC0Networks;
          string fullFN = (netIDExistsAsIS || string.IsNullOrEmpty(pathLc0Networks)) ? netDef.NetworkID : Path.Combine(CeresUserSettingsManager.Settings.DirLC0Networks, netDef.NetworkID);
          const bool VALUE_IS_LOGISTIC = false;

          NNEvaluatorOptions options = new NNEvaluatorOptions().OptionsWithOptionsDictApplied(optionsDict);

          if (!fullFN.ToUpper().Contains("ONNX"))
          {
            fullFN += ".onnx";
          }

          if (isTensorRTNative)
          {
            return NNEvaluatorTensorRT.BuildEvaluator(netDef, gpuIDs: [deviceDef.DeviceIndex], options,
                                                      ONNXNetExecutor.NetTypeEnum.LC0, fullFN);
          }
          ret = new NNEvaluatorONNX(netDef.ShortID, fullFN, null, deviceDef.Type, deviceDef.DeviceIndex, useTRT: viaTRT,
                                            ONNXNetExecutor.NetTypeEnum.LC0, maxONNXBatchSize,
                                            netDef.Precision, DEFAULT_HAS_WDL, DEFAULT_HAS_MLH,
                                            DEFAULT_HAS_UNCERTAINTYV, DEFAULT_HAS_UNCERTAINTYP, DEFAULT_HAS_ACTION,
                                            null, null, null, null, VALUE_IS_LOGISTIC, ONNX_SCALE_50_MOVE_COUNTER,
                                            false, useHistory: true, hasState: DEFAULT_HAS_STATE, options: options);
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

        case NNEvaluatorType.Ceres:
          string[] CERES_ENGINE_TYPES = { "CUDA", "CUDA16", "CUDA32",
                                          "TENSORRTNATIVE",
                                          "TENSORRT", "TENSORRT16", "TENSORRT32",
                                          "TORCHSCRIPT"};
          if (deviceDef.OverrideEngineType != null && !CERES_ENGINE_TYPES.Contains(deviceDef.OverrideEngineType.ToUpper()))
          {
            throw new Exception($"Ceres engine type not specified or invalid: {deviceDef.OverrideEngineType}."
              + System.Environment.NewLine + "Valid types: " + string.Join(", ", CERES_ENGINE_TYPES));
          }

          bool isTorchscipt = deviceDef.OverrideEngineType?.Contains("TORCHSCRIPT", StringComparison.OrdinalIgnoreCase) == true;

          // Temporary hack, Ceres nets requires positions to be retained.
          // TODO: Remove this, or make it an instance variable not global static.
          //       Leaving this globally enabled may impact performance of all evaluators.
          EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true;

          // TODO: Derive these values from NNEvaluatorOptions in the definition object
          const bool ENABLE_PROFILING = false;
          const bool USE_HISTORY = true;
          const bool HAS_UNCERTAINTY_V = true;
          const bool HAS_UNCERTAINTY_P = true;

          bool useTensorRT = false;
          bool useFP16 = true;
          if (deviceDef.Type == NNDeviceType.GPU)
          {
            // Default is CUDA 16 bit execution, but look for override.
            useTensorRT = deviceDef.OverrideEngineType != null && deviceDef.OverrideEngineType.ToUpper().StartsWith("TENSORRT");
            useFP16 = !(deviceDef.OverrideEngineType != null && deviceDef.OverrideEngineType.ToUpper().Contains("32"));
          }

          string onnxFileName = null;

          string shortID = optionsDict != null && optionsDict.TryGetValue("ID", out string id) ? id : netDef.NetworkID;
          string netFileName = onnxFileName ?? netDef.NetworkID;
          string extUpper = Path.GetExtension(netFileName).ToUpper();
          if (extUpper != ".ONNX" && extUpper != ".ENGINE" && extUpper != ".PLAN" && !isTorchscipt)
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

          NNEvaluatorOptionsCeres optionsCeres = new NNEvaluatorOptionsCeres().OptionsWithOptionsDictApplied(optionsDict) as NNEvaluatorOptionsCeres;

          int maxCeresBatchSize = useTensorRT ? TRT_MAX_BATCH_SIZE : DEFAULT_MAX_BATCH_SIZE;
          maxCeresBatchSize = deviceDef.MaxBatchSize.HasValue ? Math.Min(deviceDef.MaxBatchSize.Value, maxCeresBatchSize) : maxCeresBatchSize;

          if (isTorchscipt)
          {
            if (BuildTorchscriptEvaluator == null)
            {
              throw new Exception("NNEvaluatorFactory.BuildTorchscriptEvaluator static variable must be initialized.");
            }

            const bool USE_BFLOAT = false;
            NNEvaluator evaluatorTS = BuildTorchscriptEvaluator(netFileName, deviceDef.DeviceIndex, optionsCeres, USE_BFLOAT);
            if (evaluatorTS == null)
            {
              throw new Exception("BuildTorchscriptEvaluator returned null.");
            }

            return evaluatorTS;
          }
          else if (isTensorRTNative)
          {
            if (optionsCeres.HeadOverrides != null)
            {
              throw new NotImplementedException("Ceres TensorRT Native evaluator does not yet support head overrides.");
            }

            return NNEvaluatorTensorRT.BuildEvaluator(netDef, gpuIDs: [deviceDef.DeviceIndex], optionsCeres);
          }
          else
          {
            NNEvaluatorONNX onnxEngine = new(shortID, netFileName, null,
                                             deviceDef.Type, deviceDef.DeviceIndex, useTensorRT,
                                             ONNXNetExecutor.NetTypeEnum.TPG,
                                             maxCeresBatchSize,
                                             useFP16 ? NNEvaluatorPrecision.FP16 : NNEvaluatorPrecision.FP32,
                                             true, true, HAS_UNCERTAINTY_V, HAS_UNCERTAINTY_P, optionsCeres.UseAction,
                                             "policy", "value", "mlh", "unc", true,
                                             ENABLE_PROFILING, false, USE_HISTORY, optionsCeres,
                                             true, optionsCeres.UsePriorState, optionsCeres.HeadOverrides);

            EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true; // ** TODO: remove/rework
            onnxEngine.ConverterToFlatFromTPG = (options, o, f1) => TPGConvertersToFlat.ConvertToFlatTPGFromTPG(options, o, f1.Span);
            onnxEngine.ConverterToFlat = (options, o, history, squaresBytes, squares, legalMoveIndices)
              => TPGConvertersToFlat.ConvertToFlatTPG(options, o, history, squaresBytes, squares, legalMoveIndices);

            return onnxEngine;
          }

        case NNEvaluatorType.LC0:
          INNWeightsFileInfo net = NNWeightsFiles.LookupNetworkFile(netDef.NetworkID);
          if ((net as NNWeightsFileLC0).Format == NNWeightsFileLC0.FormatType.EmbeddedONNX)
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
                                                   1024, netDef.Precision, //netDef.Precision,
                                                   NNEvaluatorEngineTensorRTConfig.TRTPriorityLevel.Medium, null, false, TRT_SHARED);
            }
            else
            {
              // TODO: consider if we could/should delete the temporary file when
              //       it has been consumed by the ONNX engine constructor.
              // TODO: consider possibility of other precisions than FP32
              const bool USE_TRT = false;

              NNEvaluatorOptions optionsEvaluatorONNX = new NNEvaluatorOptions().OptionsWithOptionsDictApplied(optionsDict);

              //               NNEvaluatorPrecision precision = netDef.NetworkID.Contains(".16") ? NNEvaluatorPrecision.FP16 : NNEvaluatorPrecision.FP32;
              return new NNEvaluatorONNX(netDef.NetworkID, tempFN, null, deviceDef.Type, deviceDef.DeviceIndex, USE_TRT,
                                         ONNXNetExecutor.NetTypeEnum.LC0, 1024,
                                         netDef.Precision, netDefONNX.IsWDL, netDefONNX.HasMovesLeft,
                                         netDefONNX.HasUncertaintyV, netDefONNX.HasUncertaintyP, false,
                                         pbn.Net.OnnxModel.OutputValue, pbn.Net.OnnxModel.OutputWdl,
                                         pbn.Net.OnnxModel.OutputPolicy, pbn.Net.OnnxModel.OutputMlh, false, ONNX_SCALE_50_MOVE_COUNTER,
                                         false, options: optionsEvaluatorONNX);
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

            // TODO: should the NNEvaluatorCUDA constructor accept options directly so it can see them immediately?
            NNEvaluatorCUDA evaluator = new(net as NNWeightsFileLC0, deviceDef.DeviceIndex, maxBatchSize,
                                             false, netDef.Precision, false, enableCUDAGraphs, 1, referenceEvaluatorCast);
            evaluator.Options = new NNEvaluatorOptions().OptionsWithOptionsDictApplied(optionsDict);

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
          // TODO: 
          if (ComboPhasedFactory == null)
          {
            throw new Exception("NNEvaluatorFactory.ComboPhasedFactory static variable must be initialized.");
          }

          if (Custom1Options == null && optionsDict != null && optionsDict != null)
          {
            throw new NotImplementedException("Cannot apply options to ComboPhased without Custom1Options.");
          }
          NNEvaluatorOptions optionsPhased = CustomPhasedOptions ?? new NNEvaluatorOptions();
          ret = ComboPhasedFactory(netDef.NetworkID, deviceDef.DeviceIndex, referenceEvaluator, optionsPhased, optionsDict);
          break;

        case NNEvaluatorType.Custom1:
          if (Custom1Factory == null)
          {
            throw new Exception("NNEvaluatorFactory.Custom1Factory static variable must be initialized.");
          }
          if (Custom1Options == null && optionsDict != null && optionsDict != null)
          {
            throw new NotImplementedException("Cannot apply options to Custom1 without Custom1Options.");
          }
          NNEvaluatorOptions options1 = Custom1Options ?? new NNEvaluatorOptions();
          ret = Custom1Factory(netDef.NetworkID, deviceDef.DeviceIndex, referenceEvaluator, options1, optionsDict);
          break;

        case NNEvaluatorType.Custom2:
          if (Custom2Factory == null)
          {
            throw new Exception("NNEvaluatorFactory.Custom2Factory static variable must be initialized.");
          }
          if (Custom1Options == null && optionsDict != null && optionsDict != null)
          {
            throw new NotImplementedException("Cannot apply options to Custom2 without Custom1Options.");
          }
          NNEvaluatorOptions options2 = Custom2Options ?? new NNEvaluatorOptions();
          ret = Custom2Factory(netDef.NetworkID, deviceDef.DeviceIndex, referenceEvaluator, options2, optionsDict);
          break;

        default:
          throw new Exception($"Requested neural network evaluator type not supported: {netDef.Type}");
      }

      if (optionsEvaluator != null)
      {
        ret.Options = optionsEvaluator;
      }

      return ret;
    }


    static NNEvaluator BuildDeviceCombo(NNEvaluatorDef def, NNEvaluator referenceEvaluator,
                                        Dictionary<string, string> options)
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
              evaluators[i] = Singleton(def.Nets[0].Net, def.Devices[i].Device, referenceEvaluator, i, options, def.Options);
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
        NNEvaluatorDeviceComboType.Split => new NNEvaluatorSplit(evaluators, fractions, def.MinSplitNumPositions),
        NNEvaluatorDeviceComboType.RoundRobin => new NNEvaluatorRoundRobin(evaluators),
        NNEvaluatorDeviceComboType.Pooled => new NNEvaluatorPooled(evaluators),
        NNEvaluatorDeviceComboType.Compare => new NNEvaluatorCompare(evaluators),
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
        evaluators[i] = Singleton(def.Nets[i].Net, def.Devices[0].Device, referenceEvaluator, i, options, def.Options);
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
        NNEvaluatorNetComboType.Compare => new NNEvaluatorCompare(evaluators),

        NNEvaluatorNetComboType.DynamicByPos => new NNEvaluatorDynamicByPos(evaluators, def.DynamicByPosPredicate),

        _ => throw new NotImplementedException()
      };
    }


    static NNEvaluator DoBuildEvaluator(NNEvaluatorDef def, NNEvaluator referenceEvaluator, Dictionary<string, string> options)
    {
      if (def.DeviceCombo == NNEvaluatorDeviceComboType.Single && def.DeviceIndices.Length > 1)
      {
        throw new Exception("DeviceComboType.Single is not expected when number of DeviceIndices is greater than 1.");
      }

      // TODO: restore implementation, test more to see if faster or memory efficient
      const bool LC0_SERVER_ENABLE_MULTI_GPU = false; // seems somewhat slower
      if (LC0_SERVER_ENABLE_MULTI_GPU
            && def.NetCombo == NNEvaluatorNetComboType.Single
            && def.Devices.Length > 1
            && def.Devices[0].Device.Type == NNDeviceType.GPU
            && def.Nets[0].Net.Type == NNEvaluatorType.LC0)
      {
        int[] deviceIndices = new int[def.Devices.Length];
        for (int i = 0; i < def.Devices.Length; i++)
        {
          throw new NotImplementedException();
        }
      }

      if (def.DeviceCombo != NNEvaluatorDeviceComboType.Single)
      {
        // Check for special case: Split with TensorRTNative on all devices with identical nets
        // In this case, use FromDefinition which handles multi-GPU natively
        if (def.DeviceCombo == NNEvaluatorDeviceComboType.Split
            && def.Devices.Length > 0
            && def.Nets.Length > 0)
        {
          bool allTensorRTNative = def.Devices.All(d => d.Device.OverrideEngineType?.Contains("TensorRTNative", StringComparison.OrdinalIgnoreCase) == true);

          bool allNetsMatch = def.Nets.Length == 1 || def.Nets.All(n => n.Net.Equals(def.Nets[0].Net));

          if (allTensorRTNative && allNetsMatch)
          {
            def.Options = new NNEvaluatorOptionsCeres().OptionsWithOptionsDictApplied(options);
            return NNEvaluatorTensorRT.FromDefinition(def, def.Options, def.DeviceIndices);
          }
        }

        return BuildDeviceCombo(def, referenceEvaluator, options);
      }
      else if (def.NetCombo != NNEvaluatorNetComboType.Single)
      {
        return BuildNetCombo(def, referenceEvaluator, options);
      }
      else
      {
        return Singleton(def.Nets[0].Net, def.Devices[0].Device, referenceEvaluator, 0, options, def.Options);
      }
    }


  }
}
