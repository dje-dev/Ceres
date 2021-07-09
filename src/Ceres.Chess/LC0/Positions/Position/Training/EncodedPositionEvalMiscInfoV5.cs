
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
  /// Miscellaneous information associated with a position appearing in training data
  /// (binary compatible with LZ training files).
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public readonly struct EncodedPositionEvalMiscInfoV5 : IEquatable<EncodedPositionEvalMiscInfoV5>
  {
    /// <summary>
    /// Unused.
    /// </summary>
    public readonly byte DepPlyCount; 

    /// <summary>
    /// Result from our perspective.
    /// </summary>
    public readonly EncodedPositionMiscInfo.ResultCode ResultFromOurPerspective;

    /// <summary>
    /// Q at root node (from side-to-move perspective)
    /// Available only for V4 and later training data
    /// </summary>
    public readonly float RootQ;

    /// <summary>
    /// Q of best move (from side-to-move perspective)
    /// Available only for V4 and later training data
    /// </summary>
    public readonly float BestQ;

    /// <summary>
    /// D at root node (from side-to-move perspective)
    /// Available only for V4 and later training data
    /// </summary>
    public readonly float RootD;

    /// <summary>
    /// D at best move (from side-to-move perspective)
    /// Available only for V4 and later training data
    /// </summary>
    public readonly float BestD;

    /// <summary>
    /// In plies.
    /// </summary>
    public readonly float RootM;

    /// <summary>
    /// In plies.
    /// </summary>
    public readonly float BestM;

    public readonly float PliesLeft;


    public override int GetHashCode()
    {
      int part1 = ResultFromOurPerspective.GetHashCode();
      int part2 = HashCode.Combine(BestD, BestQ, RootD, BestD);

      return HashCode.Combine(part1, part2);
    }


    public override bool Equals(object obj)
    {
      if (obj is EncodedPositionEvalMiscInfoV5)
        return Equals((EncodedPositionEvalMiscInfoV5)obj);
      else
        return false;
    }


    public bool Equals(EncodedPositionEvalMiscInfoV5 other)
    {
      return  this.ResultFromOurPerspective == other.ResultFromOurPerspective
           && this.RootQ == other.RootQ
           && this.BestQ == other.BestQ
           && this.RootD == other.RootD
           && this.BestD == other.BestD;
    }

  }
}
