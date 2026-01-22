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

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// Delegate called after each sub-batch inference for tensor-major output extraction.
/// </summary>
/// <param name="globalStartPosition">Starting position index in the original batch</param>
/// <param name="positionCount">Number of actual positions processed (may be less than engineBatchSize)</param>
/// <param name="engineBatchSize">The engine's batch size (output buffer is sized for this)</param>
/// <param name="rawOutput">Raw output buffer from inference (tensor-major layout)</param>
public delegate void SubBatchOutputHandler(int globalStartPosition, int positionCount, int engineBatchSize, Half[] rawOutput);
