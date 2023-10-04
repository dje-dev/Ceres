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
  /// Represents a policy vector output by a neural network
  /// in a sparse format (indices and probabilities) rather than a dense format (1858 probabilities).
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 2)]
  public unsafe struct EncodedPolicyVectorCompressed
  {
    /// <summary>
    /// To save space, only support up to MAX_MOVES moves.
    /// Instances of positions with more moves are vanishing infrequent and will be discarded.
    /// </summary>
    public const int MAX_MOVES = 92;


    #region Raw structure data

    /// <summary>
    /// The value to be used for initialization for slots in 
    /// uncompressed dense policy vector which are not in use (typically not a legal move).
    /// </summary>
    public float FillValue;

    /// <summary>
    /// Indices of moves in policy vector.
    /// </summary>
    public fixed ushort Indices[MAX_MOVES];

    /// <summary>
    /// Probabilities of moves in policy vector.
    /// </summary>
    public fixed float Probabilities[MAX_MOVES];

    #endregion


  # region Acessor methods

    /// <summary>
    /// Span pointing to Indices.
    /// </summary>
    internal unsafe Span<ushort> IndicesSpan => MemoryMarshal.CreateSpan(ref Indices[0], MAX_MOVES);

    /// <summary>
    /// Span pointing to Probabilities.
    /// </summary>
    internal unsafe Span<float> ProbabilitiesSpan => MemoryMarshal.CreateSpan(ref Probabilities[0], MAX_MOVES);

    #endregion
  }
}