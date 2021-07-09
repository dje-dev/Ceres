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

using Ceres.Chess.EncodedPositions;
using System;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Static helper method for initializing a CompressedPolicyVector from 
  /// an unsorted Span of (move_index, probability) pairs.
  /// </summary>
  public class PolicyVectorCompressedInitializerFromProbs
  {
    [SkipLocalsInit]
    public static void InitializeFromProbsArray(ref CompressedPolicyVector policyRef, int numMoves, int numMovesToSave, Span<ProbEntry> probs)
    {
      // Due to small size insertion sort seems faster
      InsertionSort(probs, 0, numMoves - 1);
      // QuickSort(probs, 0, numMoves - 1);

      // Compute max probability so we can then
      // avoid overflow during exponentation by subtracting off
      float max = 0.0f;
      for (int j = 0; j < numMovesToSave; j++)
      {
        if (probs[j].P > max)
          max = probs[j].P;
      }

      Span<float> probsA = stackalloc float[numMoves];
      Span<int> indicesA = stackalloc int[numMoves];

      for (int j = 0; j < numMovesToSave; j++)
      {
        probsA[j] = MathF.Exp(probs[j].P - max);
        indicesA[j] = probs[j].Index;
      }

      CompressedPolicyVector.Initialize(ref policyRef, indicesA.Slice(0, numMovesToSave), probsA.Slice(0, numMovesToSave));
    }


    /// <summary>
    /// Performs partition step of a quick sort of probability entries.
    /// </summary>
    /// <param name="array"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    static int Partition(Span<ProbEntry> array, int min, int max)
    {
      float pivotP = array[max].P;
      int lowIndex = (min - 1);

      for (int j = min; j < max; j++)
      {
        if (array[j].P > pivotP)
        {
          lowIndex++;

          ProbEntry temp = array[lowIndex];
          array[lowIndex] = array[j];
          array[j] = temp;
        }
      }

      ProbEntry temp1 = array[lowIndex + 1];
      array[lowIndex + 1] = array[max];
      array[max] = temp1;

      return lowIndex + 1;
    }


    static void InsertionSort(Span<ProbEntry> array, int min, int max)
    {
      // Use unsafe code to avoid array boundary checks
      // since this is very hot code path.
      unsafe
      {
        fixed (ProbEntry* arrayPtr = &array[0])
        {
          int i = min + 1;
          while (i <= max)
          {
            ProbEntry x = arrayPtr[i];
            int j = i - 1;
            while (j >= 0 && arrayPtr[j].P < x.P)
            {
              arrayPtr[j + 1] = arrayPtr[j];
              j--;
            }
            arrayPtr[j + 1] = x;
            i++;
          }
        }
      }
    }


    /// <summary>
    /// Performs quick sort on probability entries.
    /// </summary>
    /// <param name="array"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    static void QuickSort(Span<ProbEntry> array, int min, int max)
    {
      if (min < max)
      {
        int cutpoint = Partition(array, min, max);

        QuickSort(array, min, cutpoint - 1);
        QuickSort(array, cutpoint + 1, max);
      }
    }

    public readonly struct ProbEntry
    {
      public readonly float P;
      public readonly short Index;

      public ProbEntry(short index, float p)
      {
        Index = index;
        P = p;
      }

      public override string ToString()
      {
        return $"<ProbEntry {Index} {P,6:F2}>";
      }
    }

  }
}

