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
using System.Runtime.CompilerServices;

using Ceres.Chess.NetEvaluation.Batch;

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
    public static void InitializeFromProbsArray(ref CompressedPolicyVector policyRef, 
                                                ref CompressedActionVector actions,                                               
                                                SideType side,
                                                bool sortActionsAlso,
                                                int numMoves, int numMovesToSave, Span<ProbEntry> probs)
    {
      if (sortActionsAlso)
      {
        InsertionSortWithActions(probs, ref actions, 0, numMoves - 1);
      }
      else
      {
        InsertionSort(probs, 0, numMoves - 1);
      }

      // Due to small size insertion sort seems faster.
      // QuickSort(probs, 0, numMoves - 1);

      int numToProcess = Math.Min(numMoves, numMovesToSave);

      Span<float> probsA = stackalloc float[numMoves];
      Span<int> indicesA = stackalloc int[numMoves];

      for (int j = 0; j < numToProcess; j++)
      {
        probsA[j] = probs[j].P;
        indicesA[j] = probs[j].Index;
      }

      CompressedPolicyVector.Initialize(ref policyRef, side, indicesA.Slice(0, numToProcess), probsA.Slice(0, numToProcess));
    }


    [SkipLocalsInit]
    public static void InitializeFromLogitProbsArray(ref CompressedPolicyVector policyRef, SideType side, int numMoves, int numMovesToSave, Span<ProbEntry> probs)
    {
      // Due to small size insertion sort seems faster
      InsertionSort(probs, 0, numMoves - 1);
      // QuickSort(probs, 0, numMoves - 1);

      int numToProcess = Math.Min(numMoves, numMovesToSave);

      // Compute max probability so we can then
      // avoid overflow during exponentation by subtracting off
      float max = float.MinValue;
      for (int j = 0; j < numToProcess; j++)
      {
        if (probs[j].P > max)
          max = probs[j].P;
      }

      Span<float> probsA = stackalloc float[numToProcess];
      Span<int> indicesA = stackalloc int[numToProcess];

      for (int j = 0; j < numToProcess; j++)
      {
        probsA[j] = MathF.Exp(probs[j].P - max);
        indicesA[j] = probs[j].Index;
      }

      CompressedPolicyVector.Initialize(ref policyRef, side, indicesA, probsA);
    }


    [SkipLocalsInit]
    public static void InitializeFromProbsArray(ref CompressedPolicyVector policyRef,
                                                ref CompressedActionVector actions, 
                                                SideType side,
                                                bool hasActions, bool areLogits, int numMoves, int numMovesToSave,
                                                ReadOnlySpan<Half> probs, 
                                                float cutoffMinValue = float.MinValue)
    {
      if (hasActions)
      {
        throw new NotImplementedException(); // array of actions needs to be kept sorted with the policies below
      }

      Half cutoffMinValueFloat16 = (Half)cutoffMinValue;

      // Create array of ProbEntry.
      Span<ProbEntry> probsA = stackalloc ProbEntry[numMoves];
      int numFound = 0;
      for (short i=0; i< EncodedPolicyVector.POLICY_VECTOR_LENGTH;i++)
      {
        if (probs[i] >= cutoffMinValueFloat16) // Do comparison directly with the Half, avoiding conversion unless necessary
        {
          probsA[numFound++] = new ProbEntry(i, (float)probs[i]);
        }
      }

      if (numFound == 0)
      {
        policyRef = default;
      }
      else
      {
        if (areLogits)
        {
          InitializeFromLogitProbsArray(ref policyRef, side, numFound, numMovesToSave, probsA.Slice(0, numFound));
        }
        else
        {
          CompressedActionVector dummy = default;
          InitializeFromProbsArray(ref policyRef, ref dummy, side, hasActions, numFound, numMovesToSave, probsA.Slice(0, numFound));
        }
      }
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
    /// Version of insertion sort that keeps an array of actions sorted in the same order.
    /// </summary>
    /// <param name="array"></param>
    /// <param name="actions"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    static void InsertionSortWithActions(Span<ProbEntry> array, ref CompressedActionVector actions, int min, int max)
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
            (Half W, Half L) action = actions[i];  
            int j = i - 1;
            while (j >= 0 && arrayPtr[j].P < x.P)
            {
              arrayPtr[j + 1] = arrayPtr[j];
              actions[j + 1] = actions[j];
              j--;
            }
            arrayPtr[j + 1] = x;
            actions[j + 1] = action;
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

      public override string ToString() => $"<ProbEntry {Index} {P,8:F4}>";      
    }

  }
}

