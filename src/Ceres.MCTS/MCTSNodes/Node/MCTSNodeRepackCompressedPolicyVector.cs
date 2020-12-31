#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using Directives

using Ceres.Chess.EncodedPositions;
using Ceres.MCTS.MTCSNodes.Struct;
using System;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.MCTS.MTCSNodes.Node
{
  /// <summary>
  /// Static helper class with a method that can
  /// "repack" a CompressedPolicyVector from 
  /// a node already unpacked in the tree.
  /// </summary>
  internal static class MCTSNodeRepackCompressedPolicyVector
  {
    /// <summary>
    /// Initializes a CompressedPolicyVector by extracting
    /// policy information from an MCTSNode (some of which leaves may be expanded).
    /// </summary>
    /// <param name="node"></param>
    /// <param name="policy"></param>
    [SkipLocalsInit]
    internal static void Repack(MCTSNode node, ref CompressedPolicyVector policy)
    {
      Span<ushort> indicies = stackalloc ushort[node.NumPolicyMoves];
      Span<ushort> probabilities = stackalloc ushort[node.NumPolicyMoves];

      for (int i = 0; i < node.NumPolicyMoves; i++)
      {
        ref MCTSNodeStructChild childRef = ref node.ChildAtIndexRef(i);
        indicies[i] = (ushort)childRef.Move.IndexNeuralNet;
        probabilities[i] = CompressedPolicyVector.EncodedProbability(childRef.P);
      }

      CompressedPolicyVector.Initialize(ref policy, indicies, probabilities);
    }
  }
}
