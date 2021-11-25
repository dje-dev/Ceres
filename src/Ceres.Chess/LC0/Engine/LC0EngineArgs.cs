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

using Chess.Ceres.NNEvaluators;
using System;
using System.Text;

#endregion

namespace Ceres.Chess.LC0.Engine
{
  /// <summary>
  /// Static helper methods for constructing command line arguments to 
  /// Leela Chess Zero executable.
  /// </summary>
  public static class LC0EngineArgs
  {
    public static string PrecisionString(NNEvaluatorPrecision precision)
  => precision switch
  {
    NNEvaluatorPrecision.FP16 => "cuda-fp16",
    NNEvaluatorPrecision.FP32 => "cuda",
    NNEvaluatorPrecision.Int8 => "trt-int8", // requies special build with TensorRT support
    _ => throw new Exception("Internal error: unknown precision type")
  };


    public static string BackendSyzygyArgumentString(string sygyzyPath)
    {
      return sygyzyPath == null ? "" : $" --syzygy-paths={sygyzyPath}";
    }


    public static string BackendArgumentsString(int[] gpuIDs, NNEvaluatorPrecision backend, bool fractionsAreEqual = true)
    {
      string backendName = PrecisionString(backend);

      // Default GPU ID is 0 if not specified
      if (gpuIDs == null) gpuIDs = new int[] { 0 };

      if (gpuIDs.Length == 1)
      {
        return $"--backend={backendName} --backend-opts=multi_stream=true,gpu={gpuIDs[0]} ";
      }
      else
      {
        //--backend=demux --backend-opts=(backend=cudnn-fp16,gpu=0),(backend=cudnn-fp16,gpu=1),(backend=cudnn-fp16,gpu=2),(backend=cudnn-fp16,gpu=3) --nncache=0 --movetime=-1 --nodes=1000000 -t 5
        //minimum-split-size=32,
        StringBuilder arg = new StringBuilder($"--backend={(fractionsAreEqual ? "demux" : "roundrobin")} "
                                            + $"--backend-opts=multi_stream=true,{(fractionsAreEqual ? "minimum-split-size=32," : "")}");
        for (int gpuIndex = 0; gpuIndex < gpuIDs.Length; gpuIndex++)
        {
          arg.Append($"(backend={backendName},gpu={gpuIDs[gpuIndex]}){(gpuIndex < gpuIDs.Length - 1 ? "," : " ")}");
        }

        return arg.ToString();
      }
    }

  }
}
