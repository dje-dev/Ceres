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
using Google.Protobuf.Reflection;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Mirrors the binary representation of a single training position
  /// as stored in LC0 training files (typically within compressed TAR file).
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  [Serializable]
  public readonly partial struct EncodedTrainingPosition : IEquatable<EncodedTrainingPosition>
  {
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
    /// </summary>
    public readonly EncodedPositionWithHistory Position;

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
    }

    /// <summary>
    /// Validates that key fields (game result, best Q, policies) are all valid.
    /// </summary>
    /// <param name="desc"></param>
    public void ValidateIntegrity(string desc)
    {
      ValidateWDL(desc, "BestWDL", Position.MiscInfo.InfoTraining.BestWDL);
      ValidateWDL(desc, "ResultWDL", Position.MiscInfo.InfoTraining.ResultWDL);

      float[] probs = Policies.Probabilities;
      for (int i = 0; i < 1858; i++)
      {
        
        float policy = probs[i];
        if (policy == -1)
        {
          // Invalid policiies may be represented as -1
          policy = 0;
        }

        if (float.IsNaN(policy) || policy > 1.01 || policy < 0)
        {
          throw new Exception("Policy invalid " + policy + " " + desc);
        }
      }

    }

    #endregion

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
      return Position.BoardsHistory.Equals(other.Position.BoardsHistory)
          && Version == other.Version
          && Policies.Equals(other.Policies)
          && Position.MiscInfo.Equals(other.Position.MiscInfo);          
    }

    public override int GetHashCode() => HashCode.Combine(Version, Policies, Position.BoardsHistory, Position.MiscInfo);
    

    #endregion
  }
}
