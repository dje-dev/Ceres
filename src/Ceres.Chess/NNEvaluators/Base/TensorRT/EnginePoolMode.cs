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

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// Specifies how engines in a pool handle different batch sizes.
/// </summary>
public enum EnginePoolMode
{
  /// <summary>
  /// Engines cover ranges such as [1..15], [16..127], [128..1024].
  /// </summary>
  Range,

  /// <summary>
  /// Engines for exact sizes such as 4, 16, 64, 256.
  /// </summary>
  Exact
}
