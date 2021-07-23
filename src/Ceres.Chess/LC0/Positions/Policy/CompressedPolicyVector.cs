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
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;


using Ceres.Base.Math;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Represents a policy vector, encoded in a sparse and quantized way for space efficiency.
  /// 
  /// This is currently based on the specific 1858 length policy vector used by  LeelaZero.
  /// 
  /// NOTE:
  ///   - maximum number of possible moves in a chess position believed to be about 218 but this is very artifical position
  ///   - in practice, very difficult to get even near 100 in a reasonable game of chess
  ///   - for compactness, we use an even small number (64) which almost always suffices
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 2)]
  [Serializable]
  public readonly unsafe struct CompressedPolicyVector
  {
    /// <summary>
    /// Maximum number of moves which can be encoded per position
    /// For space efficiency, we cap this at a number of moves that is very rarely exceeded
    /// </summary>
    public const int NUM_MOVE_SLOTS = 64;

    #region Raw data

    #region Move Indices

    // Not possible to make this fixed and readonly
    //public fixed ushort MoveIndices[NUM_MOVE_SLOTS];
    readonly ushort MoveIndex_0;
    readonly ushort MoveIndex_1;
    readonly ushort MoveIndex_2;
    readonly ushort MoveIndex_3;
    readonly ushort MoveIndex_4;
    readonly ushort MoveIndex_5;
    readonly ushort MoveIndex_6;
    readonly ushort MoveIndex_7;
    readonly ushort MoveIndex_8;
    readonly ushort MoveIndex_9;
    readonly ushort MoveIndex_10;
    readonly ushort MoveIndex_11;
    readonly ushort MoveIndex_12;
    readonly ushort MoveIndex_13;
    readonly ushort MoveIndex_14;
    readonly ushort MoveIndex_15;
    readonly ushort MoveIndex_16;
    readonly ushort MoveIndex_17;
    readonly ushort MoveIndex_18;
    readonly ushort MoveIndex_19;
    readonly ushort MoveIndex_20;
    readonly ushort MoveIndex_21;
    readonly ushort MoveIndex_22;
    readonly ushort MoveIndex_23;
    readonly ushort MoveIndex_24;
    readonly ushort MoveIndex_25;
    readonly ushort MoveIndex_26;
    readonly ushort MoveIndex_27;
    readonly ushort MoveIndex_28;
    readonly ushort MoveIndex_29;
    readonly ushort MoveIndex_30;
    readonly ushort MoveIndex_31;
    readonly ushort MoveIndex_32;
    readonly ushort MoveIndex_33;
    readonly ushort MoveIndex_34;
    readonly ushort MoveIndex_35;
    readonly ushort MoveIndex_36;
    readonly ushort MoveIndex_37;
    readonly ushort MoveIndex_38;
    readonly ushort MoveIndex_39;
    readonly ushort MoveIndex_40;
    readonly ushort MoveIndex_41;
    readonly ushort MoveIndex_42;
    readonly ushort MoveIndex_43;
    readonly ushort MoveIndex_44;
    readonly ushort MoveIndex_45;
    readonly ushort MoveIndex_46;
    readonly ushort MoveIndex_47;
    readonly ushort MoveIndex_48;
    readonly ushort MoveIndex_49;
    readonly ushort MoveIndex_50;
    readonly ushort MoveIndex_51;
    readonly ushort MoveIndex_52;
    readonly ushort MoveIndex_53;
    readonly ushort MoveIndex_54;
    readonly ushort MoveIndex_55;
    readonly ushort MoveIndex_56;
    readonly ushort MoveIndex_57;
    readonly ushort MoveIndex_58;
    readonly ushort MoveIndex_59;
    readonly ushort MoveIndex_60;
    readonly ushort MoveIndex_61;
    readonly ushort MoveIndex_62;
    readonly ushort MoveIndex_63;

    #endregion

    #region Move Probabilities Encoded


    // not possible in C# 7.3
    //public fixed ushort MoveProbabilitiesEncoded[NUM_MOVE_SLOTS]; // 100 / 65536

    readonly ushort MoveProbEncoded_0;
    readonly ushort MoveProbEncoded_1;
    readonly ushort MoveProbEncoded_2;
    readonly ushort MoveProbEncoded_3;
    readonly ushort MoveProbEncoded_4;
    readonly ushort MoveProbEncoded_5;
    readonly ushort MoveProbEncoded_6;
    readonly ushort MoveProbEncoded_7;
    readonly ushort MoveProbEncoded_8;
    readonly ushort MoveProbEncoded_9;
    readonly ushort MoveProbEncoded_10;
    readonly ushort MoveProbEncoded_11;
    readonly ushort MoveProbEncoded_12;
    readonly ushort MoveProbEncoded_13;
    readonly ushort MoveProbEncoded_14;
    readonly ushort MoveProbEncoded_15;
    readonly ushort MoveProbEncoded_16;
    readonly ushort MoveProbEncoded_17;
    readonly ushort MoveProbEncoded_18;
    readonly ushort MoveProbEncoded_19;
    readonly ushort MoveProbEncoded_20;
    readonly ushort MoveProbEncoded_21;
    readonly ushort MoveProbEncoded_22;
    readonly ushort MoveProbEncoded_23;
    readonly ushort MoveProbEncoded_24;
    readonly ushort MoveProbEncoded_25;
    readonly ushort MoveProbEncoded_26;
    readonly ushort MoveProbEncoded_27;
    readonly ushort MoveProbEncoded_28;
    readonly ushort MoveProbEncoded_29;
    readonly ushort MoveProbEncoded_30;
    readonly ushort MoveProbEncoded_31;
    readonly ushort MoveProbEncoded_32;
    readonly ushort MoveProbEncoded_33;
    readonly ushort MoveProbEncoded_34;
    readonly ushort MoveProbEncoded_35;
    readonly ushort MoveProbEncoded_36;
    readonly ushort MoveProbEncoded_37;
    readonly ushort MoveProbEncoded_38;
    readonly ushort MoveProbEncoded_39;
    readonly ushort MoveProbEncoded_40;
    readonly ushort MoveProbEncoded_41;
    readonly ushort MoveProbEncoded_42;
    readonly ushort MoveProbEncoded_43;
    readonly ushort MoveProbEncoded_44;
    readonly ushort MoveProbEncoded_45;
    readonly ushort MoveProbEncoded_46;
    readonly ushort MoveProbEncoded_47;
    readonly ushort MoveProbEncoded_48;
    readonly ushort MoveProbEncoded_49;
    readonly ushort MoveProbEncoded_50;
    readonly ushort MoveProbEncoded_51;
    readonly ushort MoveProbEncoded_52;
    readonly ushort MoveProbEncoded_53;
    readonly ushort MoveProbEncoded_54;
    readonly ushort MoveProbEncoded_55;
    readonly ushort MoveProbEncoded_56;
    readonly ushort MoveProbEncoded_57;
    readonly ushort MoveProbEncoded_58;
    readonly ushort MoveProbEncoded_59;
    readonly ushort MoveProbEncoded_60;
    readonly ushort MoveProbEncoded_61;
    readonly ushort MoveProbEncoded_62;
    readonly ushort MoveProbEncoded_63;

    #endregion

    #endregion

    // Not possible due to struct being readonly
    //public ReadOnlySpan<ushort> MoveProbsEncoded => MemoryMarshal.CreateReadOnlySpan(ref MoveProbEncoded_0, 1);

    #region Single float encodings

    const float HALF_INCREMENT = (float)(0.5 / 65536.0);

    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort EncodedProbability(float v)
    {
      // Rounding is simulated by an increment v of a half-step (since we end up truncating in integer space)
      v += HALF_INCREMENT; // since we truncate

      if (v >= 1.0)
        return ushort.MaxValue;
      else if (v <= 0)
        return ushort.MinValue;
      else
        return (ushort)(v * 65536.0f);
    }


    public static float DecodedProbability(ushort v) => (float)v / 65536.0f;

    #endregion

    #region Constructor

    /// <summary>
    /// Helper method to get fixed pointer to array on stack.
    /// TO DO: move to helper class
    /// </summary>
    /// <param name="probabilitiesArray"></param>
    /// <returns></returns>
    static float* Fixed(Span<float> probabilitiesArray)
    {
      unsafe
      {
        fixed (float* z = &probabilitiesArray[0])
        {
          return z;
        }
      }
    }


    /// <summary>
    /// Initializes values (bypassing readonly) with specified set of move indices and probabilities.
    /// </summary>
    /// <param name="policy"></param>
    /// <param name="indices"></param>
    /// <param name="probs"></param>
    public static void Initialize(ref CompressedPolicyVector policy, Span<ushort> indices, Span<ushort> probsEncoded)
    {
      if (indices.Length != probsEncoded.Length)
      {
        throw new ArgumentException("Length of indicies and probabilities must be same");
      }

      float lastProb = float.MaxValue;
      fixed (ushort* moveIndices = &policy.MoveIndex_0)
      {
        fixed (ushort* moveProbabilitiesEncoded = &policy.MoveProbEncoded_0)
        {
          for (int i = 0; i < indices.Length && i < NUM_MOVE_SLOTS; i++)
          {
            if (indices[i] == SPECIAL_VALUE_SENTINEL_TERMINATOR)
            {
              moveIndices[i] = SPECIAL_VALUE_SENTINEL_TERMINATOR;
              break;
            }

            moveIndices[i] = indices[i];
            moveProbabilitiesEncoded[i] = probsEncoded[i];
            Debug.Assert(DecodedProbability(probsEncoded[i]) <= lastProb);

            lastProb = probsEncoded[i];
          }
        }
      }
    }


    /// <summary>
    /// Initializes values (bypassing readonly) with specified set of move indices and probabilities.
    /// </summary>
    /// <param name="policy"></param>
    /// <param name="indices"></param>
    /// <param name="probs"></param>
    public static void Initialize(ref CompressedPolicyVector policy,
                                  Span<int> indices, Span<float> probs, bool alreadySorted = true)
    {
      // TODO: the Span<int> can actually be shortend to Span<short>

      if (indices.Length != probs.Length)
      {
        throw new ArgumentException("Length of indicies and probabilities must be same");
      }

      float probabilityAcc = 0.0f;
      float priorProb = float.MaxValue; // only used in debug mode for verifying in order
      int numMovesUsed = 0;
      fixed (ushort* moveIndices = &policy.MoveIndex_0)
      {
        fixed (ushort* moveProbabilitiesEncoded = &policy.MoveProbEncoded_0)
        {
          // Move all the probabilities into our array
          for (int i = 0; i < indices.Length && i < NUM_MOVE_SLOTS; i++)
          {
            // Get this probability and make sure is in expected sorted order
            float thisProb = probs[i];
            Debug.Assert(!alreadySorted || thisProb <= priorProb);

            // Save index
            int moveIndex = indices[i];
            moveIndices[i] = (ushort)moveIndex;

            // Save compressed probability (unless rounds to zero)
            ushort encoded = EncodedProbability(thisProb);
            if (encoded != 0)
            {
              numMovesUsed++;
              moveProbabilitiesEncoded[i] = encoded;
              probabilityAcc += thisProb;
            }
            else
            {
              break; // moves are sorted, so we will not see any more above MIN_VALUE
            }

            priorProb = thisProb;
          }

          if (numMovesUsed < NUM_MOVE_SLOTS)
          {
            // Not full. Add terminator.
            moveIndices[numMovesUsed] = SPECIAL_VALUE_SENTINEL_TERMINATOR;
          }

          // Normalize to sum to 1.0
          float adj = 1.0f / probabilityAcc;
          for (int i = 0; i < numMovesUsed; i++)
          {
            moveProbabilitiesEncoded[i] = EncodedProbability(probs[i] * adj);
          }
        }

        if (!alreadySorted) policy.Sort(numMovesUsed);
      }
    }


    /// <summary>
    /// Initializes values (bypassing readonly) with specified set of move indices and probabilities.
    /// </summary>
    /// <param name="policy"></param>
    /// <param name="probabilities"></param>
    /// <param name="alreadySorted"></param>
    public static void Initialize(ref CompressedPolicyVector policy, Span<float> probabilities, bool alreadySorted)
    {
      Initialize(ref policy, Fixed(probabilities), alreadySorted);
    }

    public const ushort SPECIAL_VALUE_SENTINEL_TERMINATOR = ushort.MaxValue;

    /// <summary>
    /// Special pseudo-value for move index indicating that a pseudorandom value is desired (with a wide policy)
    /// </summary>
    public const ushort SPECIAL_VALUE_RANDOM_WIDE = ushort.MaxValue - 1;

    /// <summary>
    /// Special pseudo-value for move index indicating that a pseudorandom value is desired (with a narrow policy)
    /// </summary>
    public const ushort SPECIAL_VALUE_RANDOM_NARROW = ushort.MaxValue - 2;

    static bool MoveIsSentinel(ushort moveRawValue) => moveRawValue >= SPECIAL_VALUE_RANDOM_NARROW;


    /// <summary>
    /// A special encoding is used to indicate if the policy is desired to be initialized
    /// randomly (for testing purposes).
    /// 
    /// The actual probabilities cannot be computed here since we don't yet know the move list,
    /// therefore weput a special value in the array to indicate that this should be expanded in subsequent processing.
    /// </summary>
    /// <param name="wide"></param>
    public static void InitializeAsRandom(ref CompressedPolicyVector policy, bool wide)
    {
      fixed (ushort* moveIndices = &policy.MoveIndex_0)
      {
        // Only need to set first move index
        moveIndices[0] = wide ? SPECIAL_VALUE_RANDOM_WIDE : SPECIAL_VALUE_RANDOM_NARROW;
      }
    }


    /// <summary>
    /// Returns the CompressedPolicyVector which is the linear combination 
    /// of a set of other raw policy vectors (using a specified set of weights).
    /// </summary>
    /// <param name="policies"></param>
    /// <param name="weights"></param>
    /// <returns></returns>
    public static CompressedPolicyVector LinearlyCombined(CompressedPolicyVector[] policies, float[] weights)
    {
      Span<float> policyAverages = stackalloc float[EncodedPolicyVector.POLICY_VECTOR_LENGTH];

      // Compute average policy result for this position
      for (int i = 0; i < policies.Length; i++)
      {
        CompressedPolicyVector policy = policies[i];
        foreach ((EncodedMove move, float probability) moveInfo in policy.ProbabilitySummary())
        {
          if (moveInfo.move.RawValue == SPECIAL_VALUE_RANDOM_NARROW ||
              moveInfo.move.RawValue == SPECIAL_VALUE_RANDOM_WIDE)
          {
            throw new NotImplementedException("Method LinearlyCombined probably not yet supported with random evaluations.");
          }

          float thisContribution = weights[i] * moveInfo.probability;
          policyAverages[moveInfo.move.IndexNeuralNet] += thisContribution;
        }
      }

      CompressedPolicyVector policyRet = default;
      Initialize(ref policyRet, policyAverages, false);
      return policyRet;
    }


    /// <summary>
    /// Returns the CompressedPolicyVector corresponding to this position when mirrored.
    /// </summary>
    public CompressedPolicyVector Mirrored
    {
      get
      {
        // TODO: improve efficiency of construction by avoiding use of temporary arrays and making two passes
        Span<ushort> indices = stackalloc ushort[CompressedPolicyVector.NUM_MOVE_SLOTS];
        Span<ushort> probs = stackalloc ushort[CompressedPolicyVector.NUM_MOVE_SLOTS];

        fixed (ushort* moveIndicesSource = &MoveIndex_0)
        fixed (ushort* moveProbsSource = &MoveProbEncoded_0)
        {
          {
            for (int i = 0; i < NUM_MOVE_SLOTS; i++)
            {
              // Tranfer probabiltiy unchanged
              probs[i] = moveProbsSource[i];

              // Transfer move mirrored (unless just a sentinel)
              if (MoveIsSentinel(moveIndicesSource[i]))
              {
                indices[i] = moveIndicesSource[i];
              }
              else
              {
                EncodedMove move = EncodedMove.FromNeuralNetIndex(moveIndicesSource[i]);// EncodedMove.FromNeuralNetIndex(moveIndicesSource[i]);
                indices[i] = (ushort)move.Mirrored.IndexNeuralNet;
              }

              // All done if we saw the terminator
              if (moveIndicesSource[i] == SPECIAL_VALUE_SENTINEL_TERMINATOR)
              {
                break;
              }
            }
          }
        }

        CompressedPolicyVector ret = new CompressedPolicyVector();
        Initialize(ref ret, indices, probs);
        return ret;
      }
    }


    /// <summary>
    /// Static method to initialize a CompressedPolicyVector from
    /// a specified array of expanded policy probabilities.
    /// </summary>
    /// <param name="policy"></param>
    /// <param name="probabilities"></param>
    /// <param name="alreadySorted"></param>
    public static void Initialize(ref CompressedPolicyVector policy, float* probabilities, bool alreadySorted)
    {
      float probabilityAcc = 0.0f;
      int numSlotsUsed = 0;
      fixed (ushort* moveIndices = &policy.MoveIndex_0)
      {
        fixed (ushort* moveProbabilitiesEncoded = &policy.MoveProbEncoded_0)
        {
          // Move all the probabilities into our array
          for (int i = 0; i < EncodedPolicyVector.POLICY_VECTOR_LENGTH; i++)
          {
            float thisProb = probabilities[i];
            probabilityAcc += thisProb;

            if (probabilities[i] > HALF_INCREMENT)
            {
              ushort encodedProb = EncodedProbability(probabilities[i]);
              if (numSlotsUsed < NUM_MOVE_SLOTS)
              {
                moveIndices[numSlotsUsed] = (ushort)i;
                moveProbabilitiesEncoded[numSlotsUsed] = encodedProb;
                //Console.WriteLine("direct " + i + " " + probabilities[i]);
                numSlotsUsed++;
              }
              else
              {
                // Find smallest index/value
                int smallestIndex = -1;
                ushort smallestValue = ushort.MaxValue;
                for (int si = 0; si < NUM_MOVE_SLOTS; si++)
                {
                  if (moveProbabilitiesEncoded[si] < smallestValue)
                  {
                    smallestIndex = si;
                    smallestValue = moveProbabilitiesEncoded[si];
                  }
                }

                ushort encodedSmallest = EncodedProbability(probabilities[smallestIndex]);
                if (moveProbabilitiesEncoded[smallestIndex] < encodedProb)
                {
                  moveIndices[smallestIndex] = (ushort)i;
                  moveProbabilitiesEncoded[smallestIndex] = encodedProb;
                }
                else
                {
                  // just drop it (lost)
                }
              }
            }

            // Add terminator if not full
            if (numSlotsUsed < NUM_MOVE_SLOTS)
            {
              moveIndices[numSlotsUsed] = SPECIAL_VALUE_SENTINEL_TERMINATOR;
            }
          }

#if DEBUG
          if (probabilityAcc < 0.995 || probabilityAcc > 1.005)
          {
            throw new Exception($"Internal error: NN probabilities sum to {probabilityAcc}");
          }
#endif
        }
      }

      if (!alreadySorted)
      {
        policy.Sort(numSlotsUsed);
      }
    }

#endregion

#region Decoding

  /// <summary>
  // Returns sum of all probabilities.
  /// </summary>
  public float SumProbabilities
  {
    get
    {
      float acc = 0;
      foreach ((EncodedMove move, float probability) in ProbabilitySummary(0, int.MaxValue))
      {
        acc += probability;
      }
      return acc;
    }
  }


  /// <summary>
  /// Returns an expanded array of all policy probabilities
  /// over all 1858 possible moves (with normalization to sum to 1.0).
  /// </summary>
  public float[] DecodedAndNormalized
    {
      get
      {
        float[] policyDecoded = DoDecoded(false);
        float acc = StatUtils.Sum(policyDecoded);
        float adj = 1.0f / acc;
        for (int j = 0; j < EncodedPolicyVector.POLICY_VECTOR_LENGTH; j++)
        {
          policyDecoded[j] *= adj;
        }

        return policyDecoded;
      }
    }

    public float[] DecodedNoValidate => DoDecoded(false);


    /// <summary>
    /// Returns an expanded array of all policy probabilities
    /// over all 1858 possible moves.
    /// </summary>
    /// <param name="validate"></param>
    /// <param name="preallocatedBuffer"></param>
    /// <returns></returns>
    public float[] DoDecoded(bool validate, float[] preallocatedBuffer = null)
    {
      if (preallocatedBuffer != null) Array.Clear(preallocatedBuffer, 0, EncodedPolicyVector.POLICY_VECTOR_LENGTH);

      fixed (ushort* moveIndices = &MoveIndex_0)
      {
        fixed (ushort* moveProbabilitiesEncoded = &MoveProbEncoded_0)
        {
          float acc = 0.0f;
          float[] ret = preallocatedBuffer ?? new float[EncodedPolicyVector.POLICY_VECTOR_LENGTH];
          for (int i = 0; i < NUM_MOVE_SLOTS; i++)
          {
            if (moveIndices[i] == SPECIAL_VALUE_SENTINEL_TERMINATOR)
            {
              break; // terminator
            }

            float decodedProb = DecodedProbability(moveProbabilitiesEncoded[i]);
            acc += decodedProb;

            // Skip moves which are just special pseudo-values
            if (moveIndices[i] == SPECIAL_VALUE_RANDOM_WIDE 
             || moveIndices[i] == SPECIAL_VALUE_RANDOM_NARROW)
            {
              continue;
            }

            Debug.Assert(moveIndices[i] < ret.Length);
            ret[moveIndices[i]] = decodedProb;
          }

          return validate && (acc < 0.99 || acc > 1.01) ? throw new Exception("Probability " + acc) 
                                                        : ret;
        }
      }

    }

#endregion

#region Helpers

    /// <summary>
    /// Returns the index of the specified move, or -1 if not found.
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    public int IndexOfMove(EncodedMove move)
    {
      ushort searchRawValue = (ushort)move.IndexNeuralNet;
      fixed (ushort* moveIndices = &MoveIndex_0)
      {
        for (int i = 0; i < NUM_MOVE_SLOTS; i++)
        {
          ushort moveRaw = moveIndices[i];
          if (moveRaw == SPECIAL_VALUE_SENTINEL_TERMINATOR)
          {
            break;
          }
          else if (moveRaw == searchRawValue)
          {
            return i;
          }
        }
        return -1;
      }
    }



    /// <summary>
    /// Reorders the entries to be sorted descending based on policy probability.
    /// TODO: make faster, or call in parallel over batch? Currently can do about 300/second batches of 1024.
    /// </summary>
    public void Sort(int numToSort)
    {
      fixed (ushort* moveIndices = &MoveIndex_0)
      {
        fixed (ushort* moveProbabilitiesEncoded = &MoveProbEncoded_0)
        {
          while (true)
          {
            // Bubble sort
            int numSwapped = 0;
            for (int i = 1; i < numToSort; i++)
            {
              if (moveProbabilitiesEncoded[i - 1] < moveProbabilitiesEncoded[i])
              {
                ushort tempI = moveIndices[i];
                ushort tempP = moveProbabilitiesEncoded[i];

                moveIndices[i] = moveIndices[i - 1];
                moveProbabilitiesEncoded[i] = moveProbabilitiesEncoded[i - 1];

                moveIndices[i - 1] = tempI;
                moveProbabilitiesEncoded[i - 1] = tempP;

                numSwapped++;
              }
            }

            if (numSwapped == 0)
            {
              return;
            }
          }
        }
      }
    }

#endregion


#region Diagnostics


    /// <summary>
    /// Compares with another CompressedPolicyVector and returns the 
    /// magnitude of the largest abolute difference in policy across all moves.
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public float MaxProbDifferenceWith(CompressedPolicyVector other)
    {
      float[] decoded = DecodedNoValidate;
      float[] otherDecoded = other.DecodedNoValidate;

      float max = 0;
      for (int i = 0; i < EncodedPolicyVector.POLICY_VECTOR_LENGTH; i++)
      {
        float diff = Math.Abs(decoded[i] - otherDecoded[i]);
        if (diff > max)
        {
          max = diff;
        }
      }
      return max;
    }


    /// <summary>
    /// Enumerates over policy moves meeting specified criteria,
    /// returning the values as Moves.
    /// </summary>
    /// <param name="startPosition"></param>
    /// <param name="minProbability"></param>
    /// <param name="topN"></param>
    /// <returns></returns>
    public IEnumerable<(Move Move, float Probability)>
      MovesAndProbabilities(Position startPosition, 
                            float minProbability = 0.0f, int topN = int.MaxValue)
    {
      MGPosition mgPos = MGPosition.FromPosition(in startPosition);
      foreach ((EncodedMove move, float probability) in ProbabilitySummary(minProbability, topN))
      {
        MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(move, in mgPos);
        Move moveRet = MGMoveConverter.ToMove(mgMove);
        yield return (moveRet, probability);
      }
    }

    /// <summary>
    /// Returns an IEnumerable over topls of moves and associated probabilites.
    /// </summary>
    /// <param name="minProbability"></param>
    /// <param name="topN"></param>
    /// <returns></returns>
    public IEnumerable<(EncodedMove Move, float Probability)> ProbabilitySummary(float minProbability = 0.0f, int topN = int.MaxValue)
    {
      (EncodedMove, float)[] moves = new (EncodedMove, float)[NUM_MOVE_SLOTS];
      int moveCount = ProbabilitySummaryList(moves, minProbability, topN);

      for (int i = 0; i < moveCount; i++)
      {
        yield return moves[i];
      }
    }


    /// <summary>
    /// Retrieves the move and associated probability at a specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public (EncodedMove Move, float Probability) PolicyInfoAtIndex(int index)
    {
      fixed (ushort* moveIndices = &MoveIndex_0)
      {
        fixed (ushort* moveProbabilitiesEncoded = &MoveProbEncoded_0)
        {
          EncodedMove move = EncodedMove.FromNeuralNetIndex(moveIndices[index]);
          float prob = DecodedProbability(moveProbabilitiesEncoded[index]);
          return (move, prob);
        }
      }
    }


    /// <summary>
    /// Populates an array of tuples with moves and associated probabilites.
    /// </summary>
    /// <param name="minProbability"></param>
    /// <returns></returns>
    public int ProbabilitySummaryList((EncodedMove, float)[] moves,
                                      float minProbability = 0.0f, int topN = int.MaxValue)
    {
      if (topN > NUM_MOVE_SLOTS) topN = NUM_MOVE_SLOTS;

      int count = 0;
      fixed (ushort* moveIndices = &MoveIndex_0)
      {
        fixed (ushort* moveProbabilitiesEncoded = &MoveProbEncoded_0)
        {
          for (int i = 0; i < topN; i++)
          {
            if (MoveIsSentinel(moveIndices[i]))
            {
              if (moveIndices[i] == SPECIAL_VALUE_SENTINEL_TERMINATOR)
              {
                break; // terminator
              }
              else if (moveIndices[i] == SPECIAL_VALUE_RANDOM_NARROW || moveIndices[i] == SPECIAL_VALUE_RANDOM_WIDE)
              {
                moves[count++] = (new EncodedMove(moveIndices[i]), float.NaN);
              }
              else
              {
                throw new Exception("Internal error, unknown sentinel.");
              }
            }
            else
            {
              float decodedProb = DecodedProbability(moveProbabilitiesEncoded[i]);

              // Break if we see a probability too small (since they are known to be sorted)
              if (decodedProb < minProbability)
              {
                break;
              }

              moves[count++] = (EncodedMove.FromNeuralNetIndex(moveIndices[i]), decodedProb);
            }
          }
        }

        return count;
      }
    }


    /// <summary>
    /// Returns short summary description string.
    /// </summary>
    /// <param name="minProbability"></param>
    /// <param name="topN"></param>
    /// <returns></returns>
    public string DumpStrShort(float minProbability = 0.0f, int topN = int.MaxValue)
    {
      string ret = "";

      var sorted = from ps in ProbabilitySummary(minProbability, topN)
                   orderby ps.Probability descending
                   select new
                   {
                     ps.Move,
                     ps.Probability
                   };

      int count = 0;
      foreach (var entry in sorted)
      {
        if (count++ >= topN)
        {
          break;
        }

        ret += " " + entry.Move.ToString() + " [" + $" {entry.Probability * 100,5:F2}%" + "] ";
      }

      return ret;
    }


    /// <summary>
    /// Dumps all moves and associated probabilities to Console.
    /// </summary>
    /// <param name="minProbability"></param>
    /// <param name="topN"></param>
    public void Dump(float minProbability = 0.0f, int topN = int.MaxValue)
    {
      foreach ((EncodedMove move, float probability) moveInfo in ProbabilitySummary(minProbability, topN))
      {
        if (moveInfo.probability > 0)
        {
          Console.WriteLine(moveInfo.move + " " + moveInfo.probability);
        }
      }
    }

#endregion


    /// <summary>
    /// Returns string summary.
    /// </summary>
    /// <returns></returns>
    public override string ToString() => DumpStrShort(0, 7);
  }

}
