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
using System.Runtime.CompilerServices;

using Ceres.Base.Benchmarking;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Static helper methods for converting between compressed and uncompressed training positions
  /// (where the policy is stored as a sparse vector).
  /// </summary>
  public static class EncodedTrainingPositionCompressedConverter
  {
    // For files written using chunking, we mark the first position of a game
    // with a sentinel value so the can be unchunked.
    public const float SENTINEL_MARK_FIRST_MOVE_IN_GAME_IN_UNUSED1 = 999;

    /// <summary>
    /// Decompresses array of compressed training positions to uncompressed training positions.
    /// </summary>
    /// <param name="source"></param>
    /// <param name="target"></param>
    /// <param name="numItems"></param>
    public static void Decompress(ReadOnlySpan<EncodedTrainingPositionCompressed> source, Span<EncodedTrainingPosition> target, int numItems)
    {
      for (int i = 0; i < numItems; ++i)
      {
        ref readonly EncodedTrainingPositionCompressed src = ref source[i];
        ref EncodedTrainingPosition dst = ref target[i];

        dst.SetVersion(src.Version);
        dst.SetInputFormat(src.InputFormat);
        DecompressPolicy(in src.Policies, ref Unsafe.AsRef(dst.Policies));
        dst.SetPositionWithBoards(in src.PositionWithBoards);

        if (dst.PositionWithBoards.MiscInfo.InfoTraining.Unused1 == EncodedTrainingPositionCompressedConverter.SENTINEL_MARK_FIRST_MOVE_IN_GAME_IN_UNUSED1)
        {
//          Console.WriteLine("sentinel at " + i);
        }
      }
    }


    /// <summary>
    /// Compresses array of uncompressed training positions to compressed training positions.
    /// </summary>
    /// <param name="source"></param>
    /// <param name="target"></param>
    public static void Compress(ReadOnlySpan<EncodedTrainingPosition> source, Span<EncodedTrainingPositionCompressed> target)
    {
      for (int i = 0; i < source.Length; ++i)
      {
        ref readonly EncodedTrainingPosition src = ref source[i];
        ref EncodedTrainingPositionCompressed dst = ref target[i];

        dst.SetVersion(src.Version);
        dst.SetInputFormat(src.InputFormat);
        dst.SetPolicies(CompressPolicy(in src.Policies));
        dst.SetPositionWithBoards(in src.PositionWithBoards);
      }

    }



    /// <summary>
    /// Converts an EncodedPolicyVector into an EncodedPolicyVectorCompressed.
    /// The new representation is lossless unless the policy vector contains more than MAX_MOVES moves.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    unsafe static EncodedPolicyVectorCompressed CompressPolicy(in EncodedPolicyVector source)
    {
      EncodedPolicyVectorCompressed dest = default;

      // Check a few slots to see if the filler value for illegal moves seems to be -1.
      bool sawNegativeOne = false;
      for (int i = 0; i < 10; i++)
      {
        if (source.ProbabilitiesPtr[i] == -1.0f)
        {
          sawNegativeOne = true;
          break;
        }
      }

      dest.FillValue = sawNegativeOne ? -1 : 0;

      // Fill indices with sentinel value.
      dest.IndicesSpan.Fill(ushort.MaxValue);

      Span<ushort> destIndicesSpan = dest.IndicesSpan;
      Span<float> destProbabilitiesSpan = dest.ProbabilitiesSpan;

      // Copy in all nonzero policies.
      int nextSlot = 0;
      for (int i=0;i<EncodedPolicyVector.POLICY_VECTOR_LENGTH;i++)
      {
        float value = source.ProbabilitiesPtr[i];
        if (value != dest.FillValue)
        {
          destIndicesSpan[nextSlot] = (ushort)i;
          destProbabilitiesSpan[nextSlot] = value;
          nextSlot++;
          if (nextSlot >= EncodedPolicyVectorCompressed.MAX_MOVES)
          {
            // No more slots available, silently drop any possible subsequent moves.
            break;
          }
        }
      }

      return dest;
    }


    /// <summary>
    /// Decompresses an EncodedPolicyVectorCompressed into an EncodedPolicyVector.
    /// </summary>
    /// <param name="source"></param>
    /// <param name="dest"></param>
    internal static unsafe void DecompressPolicy(in EncodedPolicyVectorCompressed source, ref EncodedPolicyVector dest)
    {
      // Start by filling all slots with the fill value.
      dest.ProbabilitiesSpan.Fill(source.FillValue);

      Span<ushort> sourceIndicesSpan = source.IndicesSpan;
      Span<float> sourceProbabilitiesSpan = source.ProbabilitiesSpan;
      float* probs = dest.ProbabilitiesPtr;

      for (int i = 0; i < EncodedPolicyVectorCompressed.MAX_MOVES; i++)
      {
        ushort index = sourceIndicesSpan[i];
        if (index == ushort.MaxValue)
        {
          break;
        }

        probs[index] = sourceProbabilitiesSpan[i];
      }
    }

  }
}
