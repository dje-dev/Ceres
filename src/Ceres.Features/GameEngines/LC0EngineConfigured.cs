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
using System.Collections.Concurrent;
using System.Linq;
using System.Text;
using Ceres.Chess;
using Ceres.Chess.LC0.Engine;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNFiles;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.UserSettings;
using Ceres.MCTS.Params;
using Chess.Ceres.NNEvaluators;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Helper class for constructs command line arguments for LC0,
  /// configuring as requested.
  /// </summary>
  public static class LC0EngineConfigured
  {
    /// <summary>
    /// Returns set of backend arguments to configure the
    /// backend (based net being used).
    /// </summary>
    /// <param name="evaluatorDef"></param>
    /// <returns></returns>
    static string BackendArgumentsString(NNEvaluatorDef evaluatorDef)
    {
      // LC0 does not genearlly support Int8; map into FP16 silently
      NNEvaluatorPrecision precision = evaluatorDef.Nets[0].Net.Precision;
      if (precision == NNEvaluatorPrecision.Int8)
        precision = NNEvaluatorPrecision.FP16;

      return LC0EngineArgs.BackendArgumentsString(evaluatorDef.DeviceIndices, precision, evaluatorDef.EqualFractions);
    }


    /// <summary>
    /// Returns the executable location and program arguments appropriate
    /// for LC0 given specified Ceres parameters 
    /// (optionally emulating them to the degree possible).
    /// </summary>
    /// <param name="paramsSearch"></param>
    /// <param name="paramsSelect"></param>
    /// <param name="evaluatorDef"></param>
    /// <param name="network"></param>
    /// <param name="emulateCeresOptions"></param>
    /// <param name="verboseOutput"></param>
    /// <param name="overrideEXE"></param>
    /// <param name="forceDisableSmartPruning"></param>
    /// <param name="alwaysFillHistory"></param>
    /// <returns></returns>
    public static (string, string) GetLC0EngineOptions(ParamsSearch paramsSearch,
                                                 ParamsSelect paramsSelect,
                                                 NNEvaluatorDef evaluatorDef,
                                                 INNWeightsFileInfo network,
                                                 bool emulateCeresOptions,
                                                 bool verboseOutput,
                                                 string overrideEXE = null,
                                                 bool forceDisableSmartPruning = false,
                                                 bool alwaysFillHistory = false)
    {
      if (paramsSearch == null) paramsSearch = new ParamsSearch();
      if (paramsSelect == null) paramsSelect = new ParamsSelect();

      //fail int8  string precisionStr = MCTSParams.PRECISION == WFEvalNetTensorRT.TRTPrecision.Int8 ? "trt-int8" : "cudnn-fp16";

      // Must reverse values to conform to LZ0 convention
      float FPU_MULTIPLIER = paramsSelect.FPUMode == ParamsSelect.FPUType.Reduction ? -1.0f : 1.0f;

      // TODO: plug in both versions of Centiapwn to low level Leela code
      string fpuVals = $"--fpu-value={FPU_MULTIPLIER * paramsSelect.FPUValue} --fpu-value-at-root={FPU_MULTIPLIER * paramsSelect.FPUValueAtRoot} ";
      string strategyStr = paramsSelect.FPUMode == ParamsSelect.FPUType.Absolute ? "absolute " : "reduction ";
      string fpuStr = fpuVals + "--fpu-strategy=" + strategyStr;

      string strategyAtRootStr = paramsSelect.FPUModeAtRoot == ParamsSelect.FPUType.Absolute ? "absolute " : "reduction ";
      string fpuStrRoot = paramsSelect.FPUModeAtRoot == ParamsSelect.FPUType.Same ? "--fpu-strategy-at-root=same "
                                                                          : "--fpu-strategy-at-root=" + strategyAtRootStr;

      string netSourceFile = network.FileName;

      int minibatchSize = 256; // LC0 default

      // If GPUs have equal fractions then we use demux where only 2 threads needed,
      // otherwise we use roundrobin and best is 1 + number of GPUS
      // Note that with increasing threads Leela plays significantly non-deterministically and somewhat worse
      int NUM_THREADS = evaluatorDef.EqualFractions ? 2 : evaluatorDef.Devices.Length + 1;

      string lzOptions = "--nodes-as-playouts "; // essential to get same behavior as Ceres with go nodes command 

      if (forceDisableSmartPruning || (emulateCeresOptions && !paramsSearch.FutilityPruningStopSearchEnabled))
        lzOptions += " --smart-pruning-factor=0 ";

      // Default nncache is only 200_000 but big tournaments (TCEC 19) have used as high as 20_000_000.
      // To keep memory requires reasonable for typical systems we default to a value in between.
      // However note that for very small nets such as 128x10 it may be faster to uze zero nncache.
      const int LC0_CACHE_SIZE = 5_000_000;

      // The LC0 default move-overhead is 200ms.
      // That might be appropriate for long searches (where LC0 tends to overshoot time otherwise)
      // but is too distortive/unfair to LC0 in short games.
      // Therefore we set this to 10.
      const int MOVE_OVERHEAD = 10;
      lzOptions += $"--move-overhead={MOVE_OVERHEAD}  ";
      if (alwaysFillHistory) lzOptions += $" --history-fill=always "; 

      if (emulateCeresOptions)
      {
        bool useNNCache = evaluatorDef.CacheMode > PositionEvalCache.CacheMode.None
                       || paramsSearch.Execution.TranspositionMode > TranspositionMode.None;
        int cacheSize = useNNCache ? LC0_CACHE_SIZE : 0;

        lzOptions += $@"-w {netSourceFile} -t {NUM_THREADS} --minibatch-size={minibatchSize} " +
                         //        " --policy-softmax-temp=2.2  --backend=cudnn-fp16 ";
                         $" --policy-softmax-temp={paramsSelect.PolicySoftmax} --cache-history-length={evaluatorDef.NumCacheHashPositions - 1} " +
//                         $" --score-type=win_percentage" +
                         BackendArgumentsString(evaluatorDef) +
                         //"--backend=multiplexing --backend-opts=(backend=cudnn-fp16,gpu=0),(backend=cudnn-fp16,gpu=1),(backend=cudnn-fp16,gpu=2),(backend=cudnn-fp16,gpu=3) " +
                         //                         $"--backend={precisionStr} --backend-opts=gpu={SearchParamsNN.GPU_ID_LEELA_UCI} " +

                         $"{fpuStr} {fpuStrRoot} " +
                         $" --no-sticky-endgames ";

        // + --no-out-of-order-eval"; // *** NOTE: if we add this flag, LZ0 seems to play a little different and better. TODO: study this, should we adopt?
        lzOptions += $" --cpuct-factor={paramsSelect.CPUCTFactor} --cpuct-base={paramsSelect.CPUCTBase} --cpuct={paramsSelect.CPUCT} --nncache={cacheSize} ";
        //        lzOptions += $" --max-collision-visits={paramsSearch.MAX_COLLISIONS + 1 }"; // Must increment by 1 to make comparable (also, LC0 hangs at value ot zero)
      }
      else
      {
        // Mostly we let Leela use default options, except to make it fair
        // we use a large nncache and number of threads appropriate for the number of GPUs in use
        //        lzOptions = $@"-w {weightsDir}\{netSourceFile} --minibatch-size={minibatchSize} -t {paramsNN.NNEVAL_NUM_GPUS + 1} " +
        lzOptions += $@"-w {netSourceFile} -t {NUM_THREADS} " +
//                    $"--score-type=win_percentage " +
                    $"--nncache={LC0_CACHE_SIZE} " +
                     // like TCEC 10, only 5% benefit     $"--max-prefetch=160 --max-collision-events=917 " +
                     BackendArgumentsString(evaluatorDef);

      }

      string tbPath = CeresUserSettingsManager.Settings.DirTablebases;
      if (paramsSearch.EnableTablebases) lzOptions += (@$" --syzygy-paths=#{tbPath}# ").Replace("#", "\"");

      if (verboseOutput) lzOptions += " --verbose-move-stats ";

      string EXE = CeresUserSettingsManager.GetLC0ExecutableFileName();

#if EXPERIMENTAL
      const bool LZ_USE_TRT = false; // NOTE: if true, the GPU is seemingly currently hardcoded to 3. The max batch size is 512

      if (LZ_USE_TRT)
      {
        if (network.NetworkID == "59999")
          EXE = @"C:\dev\lc0\19May\lc0\build\lc0_59999.exe";
        else if (network.NetworkID == "42767")
          EXE = @"C:\dev\lc0\19May\lc0\build\lc0_42767.exe";
        else
          throw new Exception("Unknown net for EXE " + network.NetworkID);

        if (evaluatorDef.Nets[0].Net.Precision == NNEvaluatorPrecision.Int8)
          lzOptions = lzOptions.Replace("cudnn-fp16", "trt-int8");
        else if (evaluatorDef.Nets[0].Net.Precision == NNEvaluatorPrecision.FP16)
          lzOptions = lzOptions.Replace("cudnn-fp16", "trt-fp16");
        else
          throw new NotImplementedException();
      }

#endif

      if (overrideEXE != null) EXE = overrideEXE;

      return (EXE, lzOptions);
    }


    /// <summary>
    /// Returns an LC0Engine object configured according to specified settings.
    /// </summary>
    /// <param name="paramsSearch"></param>
    /// <param name="paramsSelect"></param>
    /// <param name="evaluatorDef"></param>
    /// <param name="network"></param>
    /// <param name="resetStateAndCachesBeforeMoves"></param>
    /// <param name="emulateCeresOptions"></param>
    /// <param name="verboseOutput"></param>
    /// <param name="overrideEXE"></param>
    /// <returns></returns>
    public static LC0Engine GetLC0Engine(ParamsSearch paramsSearch,
                                         ParamsSelect paramsSelect,
                                         NNEvaluatorDef evaluatorDef,
                                         INNWeightsFileInfo network,
                                         bool resetStateAndCachesBeforeMoves,
                                         bool emulateCeresOptions,
                                         bool verboseOutput,
                                         bool forceDisableSmartPruning,
                                         string overrideEXE = null,
                                         bool alwaysFillHistory = false)
    {
      (string EXE, string lzOptions) = GetLC0EngineOptions(paramsSearch, paramsSelect, evaluatorDef, network, 
                                                           emulateCeresOptions, verboseOutput, overrideEXE, 
                                                           forceDisableSmartPruning, alwaysFillHistory);
      return new LC0Engine(EXE, lzOptions, resetStateAndCachesBeforeMoves);
    }
  }

}

