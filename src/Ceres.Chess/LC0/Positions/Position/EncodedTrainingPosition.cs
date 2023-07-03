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
using System.Runtime.InteropServices;

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
    /// Note that the actual board planes are stored in a mirrored representation
    /// compared to what needs to be fed to the neural network.
    /// </summary>
    public readonly EncodedPositionWithHistory PositionWithBoardsMirrored;

    #endregion

    #region Setters

    internal unsafe void SetVersion(int version) { fixed (int* pVersion = &Version) { *pVersion = version;  }  }
    internal unsafe void SetInputFormat(int inputFormat) { fixed (int* pInputFormat = &InputFormat) { *pInputFormat = inputFormat; } }
    internal unsafe void SetPositionWithBoardsMirrored(in EncodedPositionWithHistory pos) { fixed (EncodedPositionWithHistory* pPos = &PositionWithBoardsMirrored) { *pPos = pos; } }
    internal unsafe void SetPolicies(in EncodedPolicyVector policies) { fixed (EncodedPolicyVector* pPolicies = &Policies) { *pPolicies = policies; } }

    #endregion

    #region Validity checking


    /// <summary>
    /// Validates that a givne value is a valid probability.
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
    /// 
    /// Mostly taken from:
    ///   https://github.com/Tilps/lc0/blob/rescore_tb/src/selfplay/loop.cc#L204
    /// </summary>
    /// <param name="desc"></param>
    public void ValidateIntegrity(string desc)
    {
      // Make sure this is the supported version and input format of training data
      if (InputFormat != SUPPORTED_INPUT_FORMAT)
      {
        throw new Exception($"Found unsupported input format { InputFormat }, required is {SUPPORTED_INPUT_FORMAT}, {desc}.");
      }

      if (Version != SUPPORTED_VERSION)
      {
        throw new Exception($"Found unsupported version { Version }, required is {SUPPORTED_VERSION}, {desc}.");
      }


      int countPieces = PositionWithBoardsMirrored.BoardsHistory.History_0.CountPieces;
      if (countPieces == 0 || countPieces > 32)
      {
        throw new Exception("History board 0 has count pieces " + countPieces + ") " + desc);
      }

      ref readonly EncodedPositionEvalMiscInfoV6 refTraining = ref PositionWithBoardsMirrored.MiscInfo.InfoTraining;

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
        float[] probs = CheckPolicyValidity(desc);

        if (probs[refTraining.BestIndex] <= 0)
        {
          throw new Exception("Best policy index not positive: (" + probs[refTraining.BestIndex] + ") " + desc);
        }

        if (probs[refTraining.PlayedIndex] <= 0)
        {
          throw new Exception("Played policy index not positive: (" + probs[refTraining.PlayedIndex] + ") " + desc);
        }
      }

      if (refTraining.PliesLeft < 0)
      {
        throw new Exception("Plies left < 0 (" + refTraining.PliesLeft + ") " + desc);
      }

    }

    private float[] CheckPolicyValidity(string desc)
    {
      float[] probs = Policies.ProbabilitiesWithNegativeOneZeroed;
      float sumPolicy = 0;
      for (int i = 0; i < EncodedPolicyVector.POLICY_VECTOR_LENGTH; i++)
      {
        float policy = probs[i];
        if (policy == -1)
        {
          // Invalid policiies may be represented as -1
          policy = 0;
        }

        sumPolicy += policy;

        if (float.IsNaN(policy) || policy > 1.01 || policy < 0)
        {
          throw new Exception("Policy invalid " + policy + " " + desc);
        }
      }

      if (sumPolicy < 0.99f || sumPolicy > 1.01f)
      {
        throw new Exception("Sum policy not 1 (" + sumPolicy + ") " + desc);
      }

      return probs;
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
      EncodedPositionWithHistory thisEncPosUnmirrored = PositionWithBoardsMirrored.Mirrored;

      if (maxHistoryPositions == 2)
      {
        if (PositionWithBoardsMirrored.BoardsHistory.History_1.IsEmpty)
        {
          return new PositionWithHistory(new Position[]
          {
            Position.FromFEN(thisEncPosUnmirrored.FENForHistoryBoard(0))
          }, false);

        }
        else
        {
          return new PositionWithHistory(new Position[]
          {
            Position.FromFEN(thisEncPosUnmirrored.FENForHistoryBoard(1)),
            Position.FromFEN(thisEncPosUnmirrored.FENForHistoryBoard(0))
          }, true);
        }
      }
      else
      {
        throw new NotImplementedException(); // TODO: straightforward generalization of above needed
      }
    }

    #region Overrides

    public override bool Equals(object obj)
    {
      if (obj is EncodedTrainingPosition)
        return Equals((EncodedTrainingPosition)obj);
      else
        return false;
    }

    public bool Equals(EncodedTrainingPosition other)
    {
      return PositionWithBoardsMirrored.BoardsHistory.Equals(other.PositionWithBoardsMirrored.BoardsHistory)
          && Version == other.Version
          && Policies.Equals(other.Policies)
          && PositionWithBoardsMirrored.MiscInfo.Equals(other.PositionWithBoardsMirrored.MiscInfo);          
    }

    public override int GetHashCode() => HashCode.Combine(Version, Policies, PositionWithBoardsMirrored.BoardsHistory, PositionWithBoardsMirrored.MiscInfo);
    

    #endregion
  }
}
