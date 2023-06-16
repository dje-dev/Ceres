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

using Google.Protobuf.WellKnownTypes;
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Static helper methods for converting between compressed and uncompressed training positions
  /// (where the policy is stored as a sparse vector).
  /// </summary>
  public static class EncodedTrainingPositionCompressedConverter
  {
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
        dst.SetPositionWithBoardsMirrored(in src.PositionWithBoardsMirrored);
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
        dst.SetPositionWithBoardsMirrored(in src.PositionWithBoardsMirrored);
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

      // Fill indices with sentinel value.
      dest.IndicesSpan.Fill(ushort.MaxValue);

      // Copy in all nonzero policies.
      int nextSlot = 0;
      for (int i=0;i<EncodedPolicyVector.POLICY_VECTOR_LENGTH;i++)
      {
        float value = source.ProbabilitiesPtr[i];
        if (value > 0.0f)
        {
          dest.IndicesSpan[nextSlot] = (ushort)i;
          dest.ProbabilitiesSpan[nextSlot] = value;
          nextSlot++;
          if (nextSlot >= EncodedPolicyVectorCompressed.MAX_MOVES)
          {
            // No more slots available, silently drop any possible subsequent moves.
            break;
          }
        }
      }

      // Check a few slots to see if the filler value for illegal moves seems to be -1.
      bool sawNegativeOne = false;
      for (int i=0; i<10;i++)
      {
        if (source.ProbabilitiesPtr[i] == -1.0f)
        {
          sawNegativeOne = true;
          break;
        }
      }

      dest.FillValue = sawNegativeOne ? -1 : 0;
      return dest;
    }


    /// <summary>
    /// Decompresses an EncodedPolicyVectorCompressed into an EncodedPolicyVector.
    /// </summary>
    /// <param name="source"></param>
    /// <param name="dest"></param>
    static unsafe void DecompressPolicy(in EncodedPolicyVectorCompressed source, ref EncodedPolicyVector dest)
    {
      // Start by filling all slots with the fill value.
      // TODO: about 30% of TAR conversion time is spent here, try to improve (use FP16?).
      dest.ProbabilitiesSpan.Fill(source.FillValue);

      float* probs = dest.ProbabilitiesPtr;
      for (int i = 0; i < EncodedPolicyVectorCompressed.MAX_MOVES; i++)
      {
        if (source.IndicesSpan[i] == ushort.MaxValue)
        {
          break;
        }

        probs[source.IndicesSpan[i]] = source.ProbabilitiesSpan[i];
      }
    }

  }
}
