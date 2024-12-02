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
using Ceres.Base.DataTypes;
using Ceres.Chess.NetEvaluation.Batch;
#endregion

namespace Ceres.Chess.EncodedPositions
{
  [InlineArray(CompressedPolicyVector.NUM_MOVE_SLOTS)]
  public struct PolicyIndices
  {
    private ushort PolicyValue;
  }

  [InlineArray(CompressedPolicyVector.NUM_MOVE_SLOTS)]
  public struct PolicyValues
  {
    private ushort PolicyIndex;
  }

  /// <summary>
  /// Represents a policy vector, encoded in a sparse and quantized way for space efficiency.
  /// 
  /// This is currently based on the specific 1858 length policy vector used by  LeelaZero.
  /// 
  /// NOTE:
  ///   - maximum number of possible moves in a chess position believed to be about 218 but this is very artifical position
  ///   - in practice, very difficult to get even near 100 in a reasonable game of chess
  ///   - for compactness, we use an even small number (NUM_MOVE_SLOTS) which almost always suffices
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 2)]
  [Serializable]
  public readonly unsafe struct CompressedPolicyVector
  {
    /// <summary>
    /// Maximum number of moves which can be encoded per position
    /// For space efficiency, we cap this at a number of moves that is very rarely exceeded
    /// </summary>
    public const int NUM_MOVE_SLOTS = 80;
    
    /// <summary>
    /// In most contexts it is desirable to always use nonzero probabilities for legal moves,
    /// both to insure they are not masked out and because exactly zero probability is implausible.
    /// </summary>
    public const float DEFAULT_MIN_PROBABILITY_LEGAL_MOVE  = 0.01f * 0.05f; // 0.05%

    #region Raw data

    /// <summary>
    /// The side to play for the position.
    /// This is recorded mostly for debugging and diagnostics (to be able to translate to UCI move).
    /// </summary>
    public readonly SideType Side;

    #region Move Indices

    readonly PolicyIndices Indices;

    #endregion

    #region Move Probabilities Encoded

    readonly PolicyValues Probabilities;

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
    /// <param name="side"></param>
    /// <param name="indices"></param>
    /// <param name="probs"></param>
    public static void Initialize(ref CompressedPolicyVector policy, 
                                  SideType side, 
                                  Span<ushort> indices, 
                                  Span<ushort> probsEncoded)
    {
      if (indices.Length != probsEncoded.Length)
      {
        throw new ArgumentException("Length of indices and probabilities must be same");
      }

      fixed (SideType* sideType = &policy.Side)
      {
        *sideType = side;
      }

      float lastProb = float.MaxValue;
      fixed (ushort* moveIndices = &policy.Indices[0])
      {
        fixed (ushort* moveProbabilitiesEncoded = &policy.Probabilities[0])
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

    static CompressedActionVector dummyActionVector = default;


    /// <summary>
    /// Initializes values (bypassing readonly) with specified set of move indices and probabilities,
    /// while also keeping the actions sorted in the same order.
    /// </summary>
    /// <param name="policy"></param>
    /// <param name="side"></param>
    /// <param name="indices"></param>
    /// <param name="probs"></param>
    public static void Initialize(ref CompressedPolicyVector policy, SideType side,
                                  Span<int> indices, Span<float> probs, bool alreadySorted = true)
    {
      Initialize(ref policy, side, indices, probs, alreadySorted, false, ref dummyActionVector);
    }


    /// <summary>
    /// Initializes values (bypassing readonly) with specified set of move indices and probabilities.
    /// </summary>
    /// <param name="indices"></param>
    /// <param name="side"></param>
    /// <param name="indices"></param>
    /// <param name="probs"></param>
    public static void Initialize(ref CompressedPolicyVector policy, SideType side,
                                  Span<int> indices, Span<float> probs, 
                                  bool alreadySorted, bool withActions,
                                  ref CompressedActionVector actions)
    {
      // TODO: the Span<int> can actually be shortened to Span<short>

      if (indices.Length != probs.Length)
      {
        throw new ArgumentException("Length of indices and probabilities must be same");
      }

      fixed (SideType* sideType = &policy.Side)
      {
        *sideType = side;
      }

      float probabilityAcc = 0.0f;
      float priorProb = float.MaxValue; // only used in debug mode for verifying in order
      int numMovesUsed = 0;
      fixed (ushort* moveIndices = &policy.Indices[0])
      {
        fixed (ushort* moveProbabilitiesEncoded = &policy.Probabilities[0])
        {
          // Move all the probabilities into our array
          for (int i = 0; i < indices.Length && i < NUM_MOVE_SLOTS; i++)
          {
            // Save index
            int moveIndex = indices[i];
            if (moveIndex == SPECIAL_VALUE_SENTINEL_TERMINATOR)
            {
              break;
            }
            moveIndices[i] = (ushort)moveIndex;

            // Get this probability and make sure is in expected sorted order
            float thisProb = probs[i];
            Debug.Assert(!alreadySorted || thisProb <= priorProb);

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

        if (!alreadySorted)
        {
          if (withActions)
          {
            policy.SortWithActions(numMovesUsed, ref actions);
          }
          else
          {
            policy.Sort(numMovesUsed);
          }
        }
      }
    }


    /// <summary>
    /// Initializes values (bypassing readonly) with specified set of move indices and probabilities.
    /// </summary>
    /// <param name="policy"></param>
    /// <param name="side"></param>
    /// <param name="probabilities"></param>
    /// <param name="alreadySorted"></param>
    public static void Initialize(ref CompressedPolicyVector policy, 
                                  SideType side,
                                  Span<float> probabilities, 
                                  bool alreadySorted)
    {
      Initialize(ref policy, side, Fixed(probabilities), alreadySorted);
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
    /// therefore we put a special value in the array to indicate that this should be expanded in subsequent processing.
    /// </summary>
    /// <param name="policy"></param>
    /// <param name="side"></param>
    /// <param name="wide"></param>
    public static void InitializeAsRandom(ref CompressedPolicyVector policy, SideType side, bool wide)
    {
      fixed (ushort* moveIndices = &policy.Indices[0])
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
      Initialize(ref policyRet, policies[0].Side, policyAverages, false);
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

        
        for (int i = 0; i < NUM_MOVE_SLOTS; i++)
        {
          // Tranfer probabiltiy unchanged
          probs[i] = Probabilities[i];

          // Transfer move mirrored (unless just a sentinel)
          if (MoveIsSentinel(Indices[i]))
          {
            indices[i] = Indices[i];
          }
          else
          {
            EncodedMove move = EncodedMove.FromNeuralNetIndex(Indices[i]);// EncodedMove.FromNeuralNetIndex(moveIndicesSource[i]);
            indices[i] = (ushort)move.Mirrored.IndexNeuralNet;
          }

          // All done if we saw the terminator
          if (Indices[i] == SPECIAL_VALUE_SENTINEL_TERMINATOR)
          {
            break;
          }
        }

          CompressedPolicyVector ret = new CompressedPolicyVector();
          Initialize(ref ret, Side, indices, probs);
          return ret;
        }
      }


    /// <summary>
    /// Static method to initialize a CompressedPolicyVector from
    /// a specified array of expanded policy probabilities.
    /// </summary>
    /// <param name="policy"></param>
    /// <param name="isde"></param>
    /// <param name="probabilities"></param>
    /// <param name="alreadySorted"></param>
    public static void Initialize(ref CompressedPolicyVector policy, SideType side, float* probabilities, 
                                  bool alreadySorted, bool convertNegativeOneToZero = false)
    {
      fixed (SideType* sideType = &policy.Side)
      {
        *sideType = side;
      }

      float probabilityAcc = 0.0f;
      int numSlotsUsed = 0;
      fixed (ushort* moveIndices = &policy.Indices[0])
      {
        fixed (ushort* moveProbabilitiesEncoded = &policy.Probabilities[0])
        {
          // Move all the probabilities into our array
          for (int i = 0; i < EncodedPolicyVector.POLICY_VECTOR_LENGTH; i++)
          {
            float thisProb = probabilities[i];
            if (convertNegativeOneToZero && thisProb == -1)
            {
              thisProb = 0;
            }
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

      float acc = 0.0f;
      float[] ret = preallocatedBuffer ?? new float[EncodedPolicyVector.POLICY_VECTOR_LENGTH];
      for (int i = 0; i < NUM_MOVE_SLOTS; i++)
      {
        if (Indices[i] == SPECIAL_VALUE_SENTINEL_TERMINATOR)
        {
          break; // terminator
        }

        float decodedProb = DecodedProbability(Probabilities[i]);
        acc += decodedProb;

        // Skip moves which are just special pseudo-values
        if (Indices[i] == SPECIAL_VALUE_RANDOM_WIDE
         || Indices[i] == SPECIAL_VALUE_RANDOM_NARROW)
        {
          continue;
        }

        Debug.Assert(Indices[i] < ret.Length);
        ret[Indices[i]] = decodedProb;
      }

      return validate && (acc < 0.99 || acc > 1.01) ? throw new Exception("Probability " + acc) : ret;
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
      for (int i = 0; i < NUM_MOVE_SLOTS; i++)
      {
        ushort moveRaw = Indices[i];
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



    /// <summary>
    /// Reorders the entries to be sorted descending based on policy probability.
    /// TODO: make faster, or call in parallel over batch? Currently can do about 300/second batches of 1024.
    /// </summary>
    public void Sort(int numToSort)
    {
      fixed (ushort* moveIndices = &Indices[0])
      {
        fixed (ushort* moveProbabilitiesEncoded = &Probabilities[0])
        {
          while (true)
          {
            // Bubble sort
            int numSwapped = 0;
            for (int i = 1; i < numToSort; i++)
            {
              if (moveProbabilitiesEncoded[i - 1] < moveProbabilitiesEncoded[i])
              {
                (moveIndices[i], moveIndices[i - 1]) = (moveIndices[i - 1], moveIndices[i]);
                (moveProbabilitiesEncoded[i], moveProbabilitiesEncoded[i - 1]) = (moveProbabilitiesEncoded[i - 1], moveProbabilitiesEncoded[i]);

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


    /// <summary>
    /// Reorders the entries to be sorted descending based on policy probability.
    /// TODO: make faster, or call in parallel over batch? Currently can do about 300/second batches of 1024.
    /// </summary>
    public void SortWithActions(int numToSort, ref CompressedActionVector actions)
    {
      fixed (ushort* moveIndices = &Indices[0])
      {
        fixed (ushort* moveProbabilitiesEncoded = &Probabilities[0])
        {
          while (true)
          {
            // Bubble sort
            int numSwapped = 0;
            for (int i = 1; i < numToSort; i++)
            {
              if (moveProbabilitiesEncoded[i - 1] < moveProbabilitiesEncoded[i])
              {
                (moveIndices[i], moveIndices[i - 1]) = (moveIndices[i - 1], moveIndices[i]);
                (moveProbabilitiesEncoded[i], moveProbabilitiesEncoded[i - 1]) = (moveProbabilitiesEncoded[i - 1], moveProbabilitiesEncoded[i]);
                (actions[i], actions[i - 1]) = (actions[i - 1], actions[i]);

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
    public IEnumerable<(MGMove Move, float Probability)>
      MGMovesAndProbabilities(Position startPosition,
                            float minProbability = 0.0f, int topN = int.MaxValue)
    {
      MGPosition mgPos = MGPosition.FromPosition(in startPosition);
      foreach ((EncodedMove move, float probability) in ProbabilitySummary(minProbability, topN))
      {
        MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(move, in mgPos);
        yield return (mgMove, probability);
      }
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
    /// Returns an IEnumerable over tuples of moves and associated probabilities.
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
    /// returns the move with the highest probability.
    /// </summary>
    /// <param name="position"></param>
    /// <returns></returns>
    public MGMove TopMove(in Position position) => Count == 0 ? default : MGMovesAndProbabilities(position).ToArray()[0].Move;


    /// <summary>
    /// Retrieves the move and associated probability at a specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public (EncodedMove Move, float Probability) PolicyInfoAtIndex(int index)
    {
      int moveIndex = Indices[index];
      if (moveIndex == SPECIAL_VALUE_SENTINEL_TERMINATOR)
      {
        return (default, 0);
      }
      else
      {
        EncodedMove move = EncodedMove.FromNeuralNetIndex(Indices[index]);
        float prob = DecodedProbability(Probabilities[index]);
        return (move, prob);
      }
    }


    /// <summary>
    /// Returns the number of moves in the policy.
    /// </summary>
    public readonly int Count
    {
      get
      {
        int count = 0;
        for (int i = 0; i < NUM_MOVE_SLOTS; i++)
        {
          if (MoveIsSentinel(Indices[i]))
          {
            return count;
          }

          count++;
        }

        return count;
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
      if (topN > NUM_MOVE_SLOTS)
      {
        topN = NUM_MOVE_SLOTS;
      }

      int count = 0;
      for (int i = 0; i < topN; i++)
      {
        if (MoveIsSentinel(Indices[i]))
        {
          if (Indices[i] == SPECIAL_VALUE_SENTINEL_TERMINATOR)
          {
            break; // terminator
          }
          else if (Indices[i] == SPECIAL_VALUE_RANDOM_NARROW || Indices[i] == SPECIAL_VALUE_RANDOM_WIDE)
          {
            moves[count++] = (new EncodedMove(Indices[i]), float.NaN);
          }
          else
          {
            throw new Exception("Internal error, unknown sentinel.");
          }
        }
        else
        {
          float decodedProb = DecodedProbability(Probabilities[i]);

          // Break if we see a probability too small (since they are known to be sorted)
          if (decodedProb < minProbability)
          {
            break;
          }

          moves[count++] = (EncodedMove.FromNeuralNetIndex(Indices[i]), decodedProb);
        }
      }

      return count;

    }

    /// <summary>
    /// Returns the Shannon entropy of the policy probabilies.
    /// </summary>
    public float Entropy
    {
      get
      {
        float entropy = 0.0f;

        foreach ((_, float probability) in ProbabilitySummary())
        {
          entropy -= probability * MathF.Log(probability, 2);
        }

        return entropy;
      }
    }


    /// <summary>
    /// Returns the Kullback-Leibler divergence of this policy from another policy
    /// (this is P and vOld is Q).
    /// 
    /// TODO: Improve efficiency (loop over many items, allocate float arrays).
    /// </summary>
    /// <param name="q"></param>
    /// <returns></returns>
    public readonly float KLDWith(in CompressedPolicyVector q)
    {
      float[] decoded = this.DecodedNoValidate;
      float[] otherDecoded = q.DecodedNoValidate;

      float sum = 0;
      for (int i = 0; i < EncodedPolicyVector.POLICY_VECTOR_LENGTH; i++)
      {
        // Ensure that we avoid division by zero or taking the log of zero
        if (decoded[i] > 0 && otherDecoded[i] > 0)
        {
          sum += decoded[i] * (float)Math.Log(decoded[i] / otherDecoded[i]);
        }
      }
      return sum;
    }


    /// <summary>
    /// Returns a new policy vector with a specified temperature applied.
    /// </summary>
    /// <param name="temperature"></param>
    /// <param name="minProbability"></param>
    /// <returns></returns>
    public CompressedPolicyVector TemperatureApplied(float temperature, float minProbability = DEFAULT_MIN_PROBABILITY_LEGAL_MOVE)
    {
      if (temperature <= 0)
      {
        throw new ArgumentException("Temperature must be strictly positive.");
      }

      // First count number of moves and what new sum of exponentiated probabilites would be.
      int countMoves = 0;
      float sum = 0;
      foreach ((EncodedMove move, float probability) in ProbabilitySummary())
      {
        sum += MathF.Pow(probability, temperature);
        countMoves++;
      }

      float sumToOneMultiplier = 1.0f / sum;

      // Extract values to be used for new policy vector.
      int index = 0;
      Span<int> indices = stackalloc int[countMoves];
      Span<float> probs = stackalloc float[countMoves];
      foreach ((EncodedMove move, float probability) in ProbabilitySummary())
      {
        indices[index] = move.IndexNeuralNet;
        probs[index] = Math.Max(minProbability, MathF.Pow(probability, temperature) * sumToOneMultiplier);
        if (float.IsNaN(probs[index]))
        {
          throw new Exception("NaN probability");
        }
        index++;
      }

      // Create and return new policy vector with temperature applied.
      CompressedPolicyVector ret = new();
      Initialize(ref ret, Side, indices, probs, true);

      return ret;
    }


    /// <summary>
    /// Returns the cross entropy of the policy with a target policy.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="target"></param>
    /// <returns></returns>
    public readonly float CrossEntropyWith(in CompressedPolicyVector target)
    {
      float crossEntropy = 0.0f;
      foreach ((EncodedMove move, float probability) in ProbabilitySummary())
      {
        float targetProb = target.ProbabilityOfMove (move);
        crossEntropy += probability * MathF.Log(targetProb, 2);
      }
      return crossEntropy;
    }


    /// <summary>
    /// Returns probability of specified move according to the policy.
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    public readonly float ProbabilityOfMove(EncodedMove move) => ProbabilitySummary().Where(x => x.Move == move)
                                                                                     .Select(x => x.Probability)
                                                                                     .FirstOrDefault();     

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

        string moveStr = Side == SideType.White ? entry.Move.ToString() : entry.Move.Flipped.ToString();
        ret += " " + moveStr + " [" + $" {entry.Probability * 100,5:F2}%" + "] ";
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
