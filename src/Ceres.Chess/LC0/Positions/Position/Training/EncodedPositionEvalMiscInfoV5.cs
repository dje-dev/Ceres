
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
  public readonly record struct EncodedPositionEvalMiscInfoV5 : IEquatable<EncodedPositionEvalMiscInfoV5>
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
  }
}
