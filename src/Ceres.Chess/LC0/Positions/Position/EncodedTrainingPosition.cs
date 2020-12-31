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

using Ceres.Chess.EncodedPositions.Basic;
using System;
using System.Runtime.InteropServices;


#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Mirrors the binary representation of a single training position
  /// as stored in LC0 training files (typically within compressed TAR file).
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 2)]
  [Serializable]
  public readonly partial struct EncodedTrainingPosition : IEquatable<EncodedTrainingPosition>
  {
    // We support conversion from two versions from LC0
    // But internally we always store in V4 format (with extra fields)
    public const int V3_LEN = 8276;
    public const int V4_LEN = V3_LEN + 16;

    #region Raw structure data (Version, Policies, BoardsHistory, and MiscInfo)

    /// <summary>
    /// Version number of file.
    /// </summary>
    public readonly int Version;

    /// <summary>
    /// Policies (of length 1858 * 4 bytes).
    /// </summary>
    public readonly EncodedPolicyVector Policies;

    /// <summary>
    /// Board position (including history planes).
    /// </summary>
    public readonly EncodedPositionWithHistory Position;

    #endregion

    // Version
    // V4_STRUCT_STRING = '4s7432s832sBBBBBBBbffff'
    // V3_STRUCT_STRING = '4s7432s832sBBBBBBBb'


    /// <summary>
    /// Dumps information about the training position to the Console.
    /// </summary>
    public unsafe void Dump()
    {
      Console.WriteLine("\r\nEncodedTrainingPosition");
      Console.WriteLine ("We are " + (Position.MiscInfo.InfoPosition.SideToMove == 0 ? "White" : "Black") + " result our perspective: " + Position.MiscInfo.InfoTraining.ResultFromOurPerspective);
      Console.WriteLine("Relative points us " + Position.GetPlanesForHistoryBoard(0).RelativePointsUs);
      for (int i=0;i<8;i++) Console.WriteLine("History " + i + " " + Position.FENForHistoryBoard(i));
      for (int i = 0; i < EncodedPolicyVector.POLICY_VECTOR_LENGTH; i++)
      {
        if (Policies.Probabilities[i] != 0 && !float.IsNaN(Policies.Probabilities[i]))
        {
          bool isPawnMove = false; // TO DO: fill in 
          bool isKingMove = false; // TO DO: fill in
          EncodedMove lm = EncodedMove.FromNeuralNetIndex(i, isPawnMove, isKingMove);
        }
      }

    }

    /// <summary>
    /// Performs a few integrity checks on the position and throws Exception if any fail.
    /// </summary>
    public void CheckValid()
    {
      int size = Marshal.SizeOf(typeof(EncodedTrainingPosition));
      if (size != 8276 + 16) throw new Exception("LZTrainingPositionRaw wrong size: " + size);

      if (Position.BoardsHistory.History_0.OurKing.NumberBitsSet != 1 || Position.BoardsHistory.History_0.OurKing.NumberBitsSet != 1)
        throw new Exception("Invalid position, does not have one king per side");

      float sumProbs = Policies.SumProbabilites;
      if (sumProbs < 0.995 || sumProbs > 1.005)
        throw new Exception("Probabilities sum to " + sumProbs);
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
      return Position.BoardsHistory.Equals(other.Position.BoardsHistory)
          && Version == other.Version
          && Policies.Equals(other.Policies)
          && Position.MiscInfo.Equals(other.Position.MiscInfo);          
    }

    public override int GetHashCode() => HashCode.Combine(Version, Policies, Position.BoardsHistory, Position.MiscInfo);
    

    #endregion
  }
}
