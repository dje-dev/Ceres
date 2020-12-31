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
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Structure holding miscellaneous information
  /// that is required to be serialized as metadata 
  /// along side the raw dump of the MCTS node store (and children).
  /// </summary>
  [Serializable]
  public struct MCTSNodeStoreSerializeMiscInfo
  {
    public int VersionTag;

    public string Description;

    public MCTSNodeStructIndex RootIndex;
    public PositionWithHistory PriorMoves;

    public int NumNodesReserved;
    public int NumNodesAllocated;
    public long NumChildrenAllocated;
  }
}