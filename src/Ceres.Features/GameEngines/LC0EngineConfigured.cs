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
using System.Diagnostics.SymbolStore;
using System.Linq;
using System.Text;
using Ceres.Base.OperatingSystem;
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
    public static int? OVERRIDE_LC0_BATCH_SIZE = null;


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
    /// <param name="overrideBatchSize"></param>
    /// <param name="overrideCacheSize"></param>
    /// <returns></returns>
    public static (string, string) GetLC0EngineOptions(ParamsSearch paramsSearch,
                                                 ParamsSelect paramsSelect,
                                                 NNEvaluatorDef evaluatorDef,
                                                 INNWeightsFileInfo network,
                                                 bool emulateCeresOptions,
                                                 bool verboseOutput,
                                                 string overrideEXE = null,
                                                 bool forceDisableSmartPruning = false,
                                                 bool alwaysFillHistory = false,
                                                 int? overrideBatchSize = null,
                                                 int? overrideCacheSize = null)
    {
      if (paramsSearch == null)
      {
        paramsSearch = new ParamsSearch();
      }

      if (paramsSelect == null)
      {
        paramsSelect = new ParamsSelect();
      }
      //fail int8  string precisionStr = MCTSParams.PRECISION == WFEvalNetTensorRT.TRTPrecision.Int8 ? "trt-int8" : "cudnn-fp16";

      string netSourceFile = network.FileName;

      // If GPUs have equal fractions then we use demux where only 2 threads needed,
      // otherwise we use roundrobin and best is 1 + number of GPUS
      // Note that with increasing threads Leela plays significantly non-deterministically and somewhat worse
      int NUM_THREADS = (evaluatorDef == null || evaluatorDef.EqualFractions) ? 2 : evaluatorDef.Devices.Length + 1;
      
      string lzOptions = "--nodes-as-playouts "; // essential to get same behavior as Ceres with go nodes command 

#if NOT
      if (paramsSearch.TestFlag)
      {
        // Turn off MLH
        lzOptions += " --moves-left-max-effect=0 --moves-left-threshold=1 --moves-left-scaled-factor=0 --moves-left-quadratic-factor=0 ";
      }
#endif

      if (forceDisableSmartPruning || (emulateCeresOptions && !paramsSearch.FutilityPruningStopSearchEnabled))
      {
        lzOptions += " --smart-pruning-factor=0 ";
      }


      int MOVE_OVERHEAD = (int)(new ParamsSearch().MoveOverheadSeconds * 1000);
      lzOptions += $"--move-overhead={MOVE_OVERHEAD} ";

      if (alwaysFillHistory) lzOptions += $" --history-fill=always "; 

      if (overrideBatchSize == null && OVERRIDE_LC0_BATCH_SIZE.HasValue)
      {
        overrideBatchSize = OVERRIDE_LC0_BATCH_SIZE;
      }

      if (overrideBatchSize != null)
      {
        lzOptions += $"--minibatch-size={overrideBatchSize} ";
      }

      // LC0 speed generally much improved by a cache size larger than the default (200_000).
      const float LC0_CACHE_FRAC_MEM = 0.05f; // sufficiently small to allow multiple concurrent engines
      const int BYTES_PER_CACHE_ITEM = 250;
      int DEFAULT_CACHE_SIZE= (int)Math.Max(2_000_000, (HardwareManager.MemorySize  * LC0_CACHE_FRAC_MEM) / BYTES_PER_CACHE_ITEM);
      int cacheSize = overrideCacheSize ?? DEFAULT_CACHE_SIZE;
      lzOptions += $"--nncache={cacheSize} ";

      if (emulateCeresOptions)
      {
        throw new NotImplementedException();
#if NOT
        // Must reverse values to conform to LZ0 convention
        float FPU_MULTIPLIER = paramsSelect.FPUMode == ParamsSelect.FPUType.Reduction ? -1.0f : 1.0f;

        // TODO: plug in both versions of Centiapwn to low level Leela code
        string fpuVals = $"--fpu-value={FPU_MULTIPLIER * paramsSelect.FPUValue} --fpu-value-at-root={FPU_MULTIPLIER * paramsSelect.FPUValueAtRoot} ";
        string strategyStr = paramsSelect.FPUMode == ParamsSelect.FPUType.Absolute ? "absolute " : "reduction ";
        string fpuStr = fpuVals + "--fpu-strategy=" + strategyStr;

        string strategyAtRootStr = paramsSelect.FPUModeAtRoot == ParamsSelect.FPUType.Absolute ? "absolute " : "reduction ";
        string fpuStrRoot = paramsSelect.FPUModeAtRoot == ParamsSelect.FPUType.Same ? "--fpu-strategy-at-root=same "
                                                                            : "--fpu-strategy-at-root=" + strategyAtRootStr;

        bool useNNCache = evaluatorDef.CacheMode > PositionEvalCache.CacheMode.None
                       || paramsSearch.Execution.TranspositionMode > TranspositionMode.None;
        int cacheSize = 0; //useNNCache ? LC0_CACHE_SIZE : 0;

        lzOptions += $@"-w {netSourceFile} -t {NUM_THREADS} " +
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
#endif
      }
      else
      {
        // Mostly we let Leela use default options, except to make it fair
        // we use a large nncache and number of threads appropriate for the number of GPUs in use
        //        lzOptions = $@"-w {weightsDir}\{netSourceFile} --minibatch-size={minibatchSize} -t {paramsNN.NNEVAL_NUM_GPUS + 1} " +
        lzOptions += $@"-w {netSourceFile} -t {NUM_THREADS} " +
//                    $"--score-type=win_percentage " +
                     // like TCEC 10, only 5% benefit     $"--max-prefetch=160 --max-collision-events=917 " +
                     BackendArgumentsString(evaluatorDef);

      }

      string tbPath = CeresUserSettingsManager.Settings.TablebaseDirectory;
      if (paramsSearch.EnableTablebases)
      {
        lzOptions += (@$" --syzygy-paths=#{tbPath}# ").Replace("#", "\"");
      }

      if (verboseOutput)
      {
        lzOptions += " --verbose-move-stats ";
      }

      string EXE = CeresUserSettingsManager.GetLC0ExecutableFileName();

      if (overrideEXE != null)
      {
        EXE = overrideEXE;
      }

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
    /// <param name="forceDisableSmartPruning"></param>
    /// <param name="overrideEXE"></param>
    /// <param name="alwaysFillHistory"></param>
    /// <param name="extraCommandLineArgs"></param>
    /// <param name="overrideBatchSize"></param>
    /// <param name="overrideCacheSize"></param>
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
                                         bool alwaysFillHistory = false,
                                         string extraCommandLineArgs = null,
                                         int? overrideBatchSize = null,
                                         int? overrideCacheSize = null)
    {
      (string EXE, string lzOptions) = GetLC0EngineOptions(paramsSearch, paramsSelect, evaluatorDef, network, 
                                                           emulateCeresOptions, verboseOutput, overrideEXE, 
                                                           forceDisableSmartPruning, alwaysFillHistory, 
                                                           overrideBatchSize, overrideCacheSize);
      if (extraCommandLineArgs != null) lzOptions += " " + extraCommandLineArgs;
      return new LC0Engine(EXE, lzOptions, resetStateAndCachesBeforeMoves);
    }


    /// <summary>
    /// Returns set of backend arguments to configure the
    /// backend (based net being used).
    /// </summary>
    /// <param name="evaluatorDef"></param>
    /// <returns></returns>
    static string BackendArgumentsString(NNEvaluatorDef evaluatorDef)
    {
      if (evaluatorDef == null)
      {
        return "";
      }
      else
      {
        // LC0 does not genearlly support Int8; map into FP16 silently
        NNEvaluatorPrecision precision = evaluatorDef.Nets[0].Net.Precision;
        if (precision == NNEvaluatorPrecision.Int8)
        {
          precision = NNEvaluatorPrecision.FP16;
        }

      return LC0EngineArgs.BackendArgumentsString(evaluatorDef.DeviceIndices, precision, evaluatorDef.EqualFractions);
      }
    }


  }

}

