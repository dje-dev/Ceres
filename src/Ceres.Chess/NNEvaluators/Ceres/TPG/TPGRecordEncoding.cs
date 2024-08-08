#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/cerestrain.
  Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with CeresTrain. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;

#endregion

namespace Ceres.Chess.NNEvaluators.Ceres.TPG
{
  /// <summary>
  /// Static helper methods relating to encoding/decoding of TPG records.
  /// 
  /// In particular, some fields (such as MLH value) have to be scaled to
  /// fit into required range (e.g. one byte).
  /// </summary>
  public static class TPGRecordEncoding
  {
    /// <summary>
    /// If experimental feature to include the WDL from prior position in the TPGRecord is enabled.
    /// 
    /// TODO: REMOVE THIS
    /// </summary>
    public const bool ENABLE_PRIOR_VALUE_POSITION = false;

    // Scaling factor by which the win/draw/loss probabilities
    // from the prior board position are multiplied.
    // We expand the range of values to [0, 2.0] to reduce quantization error
    // (to use increments of 0.005).
    internal const float PRIOR_POS_VALUE_PROB_MULTIPLIER = 2.0f;

    internal static int Move50CountDecoded(float mlh)
    {
      const float MOVE_50_DIVISOR = 50;
      return (int)Math.Round(mlh * MOVE_50_DIVISOR);
    }

    internal static float Move50CountEncoded(int move50Count)
    {
      // With new encoding, put into a range [0, 2.0].
      const float MOVE_50_DIVISOR = 50;
      return Math.Min(move50Count, 100) / MOVE_50_DIVISOR;
    }


    // Unclear what value to use if square has never seen a move.
    // Don't want to use small value indicating move was recently made.
    // Don't want to use a very large value suggesting some sort of fortress.
    // Therefore pick something intermediate.
    internal const byte DEFAULT_PLIES_SINCE_LAST_PIECE_MOVED_IF_STARTPOS = 30;

    public static byte ToPliesSinceLastPieceMoveBySquare(int indexMoveInGame, int indexLastMoveThisSquare)
    {
      if (indexLastMoveThisSquare == 0) // never moved
      {
        return (byte)Math.Max(DEFAULT_PLIES_SINCE_LAST_PIECE_MOVED_IF_STARTPOS, indexMoveInGame);
      }
      else
      {
        int movesDiff = Math.Min(byte.MaxValue, 1 + indexMoveInGame - indexLastMoveThisSquare);
        return (byte)movesDiff;

      }
    }

    internal const int PlySinceLastMoveValueIfWasLastMove = 2;

    internal static float PliesSinceLastMoveEncoded(int numPlies)
    {
      numPlies = Math.Min(numPlies, byte.MaxValue);

      // Take square root so somewhat big difference
      // between lower values such as 0 and 1.
      // Then scale so max value is 2.
      //        return MathF.Sqrt(numPlies) / 5;
      return PlySinceLastMoveValueIfWasLastMove / MathF.Sqrt(numPlies + 1);
    }

    internal const float MAX_MLH = byte.MaxValue;

    /// <summary>
    /// Scale MLH targets to be closer to values for other target variables.
    /// </summary>
    const float MLH_SCALING_FACTOR = 0.1f;
    public static float MLHEncoded(float mlh)
    {
      if (mlh < 0 || float.IsNaN(mlh))
      {
        mlh = 10; // rare bad data condition, fill in with arbitrary valid value
      }

      return MLH_SCALING_FACTOR * MathF.Sqrt(Math.Max(0, Math.Min(MAX_MLH, mlh)));
    }

    public static float MLHDecoded(float mlh) => MathF.Pow(mlh / MLH_SCALING_FACTOR, 2);
  }
}
