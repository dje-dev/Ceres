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


#endregion

namespace Ceres.Chess.NNEvaluators.Defs
{
  public enum NNEvaluatorType
  {
    /// <summary>
    /// Leela Zero network, run via slightly modified LC0 library.
    /// </summary>
    LC0Library,

    /// <summary>
    /// In ONNX format to be run via ONNX runtime
    /// </summary>
    ONNXViaORT,

    /// <summary>
    /// In ONNX format to be run via TensorRT
    /// </summary>
    ONNXViaTRT,

    /// <summary>
    /// Random policy and value, using a relatively wide policy distribution (leading to wide search trees)
    /// </summary>
    RandomWide,

    /// <summary>
    /// Random policy and value, using a relatively narrow policy distribution (leading to deep search trees)
    /// </summary>
    RandomNarrow,

    /// <summary>
    /// One evaluator used in early phase of search tree, second evalutor for remainder.
    /// </summary>
    ComboPhased,

    /// <summary>
    /// A custom evaluator installed into NNEvaluatorDefFactory.Custom1Factory.
    /// </summary>
    Custom1,

    /// <summary>
    /// A custom evaluator installed into NNEvaluatorDefFactory.Custom2Factory.
    /// </summary>
    Custom2
  };
}
