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
  /// Structure just like EncodedTrainingPosition except that 
  /// only policy is not the very sparse array of length 1858 
  /// but instead an EncodedPolicyVectorCompressed.
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  [Serializable]
  public readonly partial struct EncodedTrainingPositionCompressed
  {
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
    /// Compressed policy vector.
    /// </summary>
    public readonly EncodedPolicyVectorCompressed Policies;

    /// <summary>
    /// Board position (including history planes).
    /// Note that the actual board planes are stored in a mirrored representation
    /// compared to what needs to be fed to the neural network.
    /// </summary>
    public readonly EncodedPositionWithHistory PositionWithBoards;

    #endregion


    #region Setters

    internal unsafe void SetVersion(int version) { fixed (int* pVersion = &Version) { *pVersion = version; } }
    internal unsafe void SetInputFormat(int inputFormat) { fixed (int* pInputFormat = &InputFormat) { *pInputFormat = inputFormat; } }
    internal unsafe void SetPositionWithBoards(in EncodedPositionWithHistory pos) { fixed (EncodedPositionWithHistory* pPos = &PositionWithBoards) { *pPos = pos; } }
    internal unsafe void SetPolicies(in EncodedPolicyVectorCompressed policies) { fixed (EncodedPolicyVectorCompressed* pPolicies = &Policies) { *pPolicies = policies; } }

    #endregion

  }
}
