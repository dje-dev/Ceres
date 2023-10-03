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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.LC0.Boards;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Positions;


#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Identical binary representation of a single training position
  /// as stored in LC0 training files (typically within compressed TAR file).
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  [Serializable]
  public readonly partial struct EncodedTrainingPosition : IEquatable<EncodedTrainingPosition>
  {
    // Only v6 data using input format 1 is supported.
    // This keeps the code simple and allows unrestricted use of advanced v6 features.
    public const int SUPPORTED_VERSION = 6;
    public const int SUPPORTED_INPUT_FORMAT = 1;

    public const int V4_LEN = 8276 + 16;
    public const int V5_LEN = 8308;
    public const int V6_LEN = 8356;

    #region Raw structure data (Version, Policies, BoardsHistory, and MiscInfo)

    /// <summary>
    /// Version number of file.
    /// </summary>
    public readonly int Version;

    /// <summary>
    ///  Board representation input format.
    /// </summary>
    public readonly int InputFormat;

    /// <summary>
    /// Policies (of length 1858 * 4 bytes).
    /// </summary>
    public readonly EncodedPolicyVector Policies;


    /// <summary>
    /// Board position (including history planes).
    /// Note that these LC0 training data files (TARs) contain mirrored positions
    /// (compared to the representation that must be fed to the LC0 neural network).
    /// However this mirroring is undone immediately after reading from disk and
    /// this field in memory is always in the natural representation.
    /// </summary>
    public readonly EncodedPositionWithHistory PositionWithBoards;

    #endregion

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="version"></param>
    /// <param name="inputFormat"></param>
    /// <param name="pos"></param>
    /// <param name="policy"></param>
    public EncodedTrainingPosition(int version, int inputFormat, in EncodedPositionWithHistory pos, in EncodedPolicyVector policy)
    {
      Version = version;
      InputFormat = inputFormat;
      PositionWithBoards = pos;
      Policies = policy;
    }

    #region Setters

    internal unsafe readonly void SetVersion(int version) { fixed (int* pVersion = &Version) { *pVersion = version;  }  }
    internal unsafe readonly void SetInputFormat(int inputFormat) { fixed (int* pInputFormat = &InputFormat) { *pInputFormat = inputFormat; } }
    internal unsafe readonly void SetPositionWithBoards(in EncodedPositionWithHistory pos) { fixed (EncodedPositionWithHistory* pPos = &PositionWithBoards) { *pPos = pos; } }
    internal unsafe readonly void SetPolicies(in EncodedPolicyVector policies) { fixed (EncodedPolicyVector* pPolicies = &Policies) { *pPolicies = policies; } }

    #endregion



    /// <summary>
    /// Returns a Position which for the current move (last history position).
    /// </summary>
    /// <returns></returns>
    public readonly Position FinalPosition =>PositionWithBoards.HistoryPosition(0);


    #region Mirroring

    /// <summary>
    /// LC0 training files (TAR) contain mirrored boards. We want to hide this implementation detail/annoyance.
    /// This method mirrors all the boards (to put them into the natural representation)
    /// and should be called immediately after either:
    ///   - we read from disk, or
    ///   - we are about to write back to disk.
    ///
    /// Note that the method is marked as readonly (to prevent defensive copies), but this is not true.
    /// 
    /// N.B.: The policy vector is stored mirrored, and unlike the move indices and board handled here,
    ///       we currently do not undo this mirroring. 
    ///       TODO: Someday improve this, and unmirror/mirror upon load/save from disk only 
    ///             for consistency and efficiency and clarity.
    ///             However this requires extensive retesting, and CompressedPolicyVector likely needs related changes.f
    /// </summary>
    /// <param name="boards"></param>
    /// <param name="numItems"></param>
    public readonly void MirrorInPlace()
    {
      // Mirror the boards themselves.
      PositionWithBoards.BoardsHistory.MirrorBoardsInPlace();

      // Any recorded move indices have to be mirrored as well.
      short playedIndexMirrored = (short)PositionWithBoards.MiscInfo.InfoTraining.PlayedMove.Mirrored.IndexNeuralNet;
      short bestIndexMirrored = (short)PositionWithBoards.MiscInfo.InfoTraining.BestMove.Mirrored.IndexNeuralNet;
      PositionWithBoards.MiscInfo.InfoTraining.SetPlayedAndBestIndex(playedIndexMirrored, bestIndexMirrored);
    }


    /// <summary>
    /// Mirrors all positions (in place) within a span of EncodedTrainingPosition objects, up to specified number of items.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="numItems"></param>
    public static void MirrorPositions(Span<EncodedTrainingPosition> positions, int numItems)
    {
      for (int i = 0; i < numItems; i++)
      {
        positions[i].MirrorInPlace();
      }
    }

#endregion

    #region Validity checking

    /// <summary>
    /// Validates that a given value is a valid probability.
    /// </summary>
    /// <param name="desc1"></param>
    /// <param name="desc2"></param>
    /// <param name="prob"></param>
    static void ValidateProbability(string desc1, string desc2, float prob)
    {
      if (float.IsNaN(prob))
      {
        throw new Exception("NaN probability found " + desc1 + " " + desc2);
      }

      if (prob < -1.01f || prob > 1.01f)
      {
        throw new Exception("Probability outside range [-1.01, 1.01] " + desc1 + " " + desc2);
      }
    }

    /// <summary>
    /// Validates that all values in a WDL triple are valid probabilities.
    /// </summary>
    /// <param name="desc1"></param>
    /// <param name="desc2"></param>
    /// <param name="wdl"></param>
    static void ValidateWDL(string desc1, string desc2, (float w, float d, float l) wdl)
    {
      ValidateProbability(desc1, desc2, wdl.w);
      ValidateProbability(desc1, desc2, wdl.d);
      ValidateProbability(desc1, desc2, wdl.l);

      float sum = wdl.w + wdl.d + wdl.l;
      if (sum < 0.99f || sum > 1.01f)
      {
        throw new Exception("Sum probabilities not 1" + desc1 + " " + desc2);
      }
    }

    static bool haveWarnedBestQ = false;


    /// <summary>
    /// Validates that key fields (game result, best Q, policies, etc.) are all valid.
    /// <param name="desc"></param>
    public readonly void ValidateIntegrity(string desc)
    {
      ValidateIntegrity(InputFormat, Version, in PositionWithBoards, in Policies, desc);
    }


    /// <summary>
    /// Validates that specified key fields (game result, best Q, policies, etc.) are all valid.
    /// 
    /// Mostly taken from:
    ///   https://github.com/Tilps/lc0/blob/rescore_tb/src/selfplay/loop.cc#L204
    /// </summary>
    /// <param name="desc"></param>
    public static void ValidateIntegrity(int inputFormat, int version,
                                         in EncodedPositionWithHistory boardsHistory, 
                                         in EncodedPolicyVector policyVector, 
                                         string desc)
    {
      // Make sure this is the supported version and input format of training data
      if (inputFormat != SUPPORTED_INPUT_FORMAT)
      {
        throw new Exception($"Found unsupported input format { inputFormat }, required is {SUPPORTED_INPUT_FORMAT}, {desc}.");
      }

      if (version != SUPPORTED_VERSION)
      {
        throw new Exception($"Found unsupported version { version }, required is {SUPPORTED_VERSION}, {desc}.");
      }


      int countPieces = boardsHistory.BoardsHistory.History_0.CountPieces;
      if (countPieces == 0 || countPieces > 32)
      {
        throw new Exception("History board 0 has count pieces " + countPieces + ") " + desc);
      }

      ref readonly EncodedPositionEvalMiscInfoV6 refTraining = ref boardsHistory.MiscInfo.InfoTraining;

      if (float.IsNaN(refTraining.BestD + refTraining.BestQ))
      {
        if (!haveWarnedBestQ)
        {
          Console.WriteLine("WARNING: BestD or BestQ is NaN. Omit subsequent warnings of this type.");
          haveWarnedBestQ = true;
        }
      }

      if (float.IsNaN(refTraining.ResultD + refTraining.ResultQ))
      {
        throw new Exception("ResultD or ResultQ is NaN");
      }

      if (!float.IsNaN(refTraining.OriginalM) && refTraining.OriginalM < 0)
      {
        throw new Exception("OriginalM < 0 (" + refTraining.OriginalM + ") " + desc);
      }

      if (refTraining.ResultD == 0 && refTraining.ResultQ == 0)
      {
        throw new Exception("Both ResultD and ResultQ are zero. " + desc);
      }

      if (refTraining.BestD == 0 && refTraining.BestQ == 0)
      {
        throw new Exception("Both BestD and BestQ are zero. " + desc);
      }

      ValidateWDL(desc, "BestWDL", refTraining.BestWDL);
      ValidateWDL(desc, "ResultWDL", refTraining.ResultWDL);

#if DEBUG
      const bool validate = true;
#else
      // Computationally expensive, randomly validate only a few in non-debug mode.
      const int VALIDATE_PCT = 1;
      bool validate = (refTraining.NumVisits % 100) < VALIDATE_PCT;
#endif
      if (validate)
      {
        float[] probs = policyVector.CheckPolicyValidity(desc);

        if (probs[refTraining.BestMove.Mirrored.IndexNeuralNet] <= 0)
        {
          throw new Exception("Best policy index not positive: (" + probs[refTraining.BestIndex] + ") " + desc);
        }

        if (probs[refTraining.PlayedMove.Mirrored.IndexNeuralNet] <= 0)
        {
          throw new Exception("Played policy index not positive: (" + probs[refTraining.PlayedIndex] + ") " + desc);
        }
      }

      if (refTraining.PliesLeft < 0)
      {
        throw new Exception("Plies left < 0 (" + refTraining.PliesLeft + ") " + desc);
      }

    }

    #endregion


    /// <summary>
    /// Converts to a PositionWithHistory object.
    /// </summary>
    /// <param name="maxHistoryPositions"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public readonly PositionWithHistory ToPositionWithHistory(int maxHistoryPositions)
    {
      int numAdded = 0;
      Span<Position> positions = stackalloc Position[maxHistoryPositions];
      for (int i = maxHistoryPositions - 1; i >= 0; i--)
      {
        if (!PositionWithBoards.GetPlanesForHistoryBoard(i).IsEmpty)
        {
          positions[numAdded++] = PositionWithBoards.HistoryPosition(i);
        }
      }

      // First position may be incorrect (missing en passant)
      // since the prior history move not available to detect.
      // TODO: Try to infer this from the move actually played.
      const bool EARLIEST_POSITION_MAY_BE_MISSING_EN_PASSANT = true;
      return new PositionWithHistory(positions.Slice(0, numAdded), EARLIEST_POSITION_MAY_BE_MISSING_EN_PASSANT, false); ;
    }

    #region Overrides

    public readonly override bool Equals(object obj)
    {
      if (obj is EncodedTrainingPosition)
        return Equals((EncodedTrainingPosition)obj);
      else
        return false;
    }

    public readonly bool Equals(EncodedTrainingPosition other)
    {
      return PositionWithBoards.BoardsHistory.Equals(other.PositionWithBoards.BoardsHistory)
          && Version == other.Version
          && Policies.Equals(other.Policies)
          && PositionWithBoards.MiscInfo.Equals(other.PositionWithBoards.MiscInfo);          
    }

    public readonly override int GetHashCode() => HashCode.Combine(Version, Policies, PositionWithBoards.BoardsHistory, PositionWithBoards.MiscInfo);
    

    #endregion
  }
}
