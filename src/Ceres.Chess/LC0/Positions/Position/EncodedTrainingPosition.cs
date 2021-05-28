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
