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
/// Output tensor descriptor with name and data location.
/// </summary>
/// <param name="Name">Name of the output tensor.</param>
/// <param name="Offset">Offset in elements from start of output buffer.</param>
/// <param name="Size">Size in elements.</param>
public readonly record struct OutputTensorInfo(string Name, long Offset, long Size);
