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
using Ceres.MCTS.MTCSNodes;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// If a node is tranposition linked and more than just the first
  /// evaluation has been extracted from the root, then a visitor
  /// is created to sequentially visit the nodes under this node.
  /// 
  /// This class contains that visitor, and also records the size
  /// of that subtree at the time the visitor is created (since the visitor
  /// can only guarantee to visit nodes already existing in the transposition
  /// subtree at the time the visitor is created).
  /// </summary>
  public class MCTSNodeTranspositionVisitor : ICloneable
  {
    /// <summary>
    /// Record the N of the transposition at the time that the visitor was created
    /// </summary>
    public int TranspositionRootNWhenVisitsStarted;

    /// <summary>
    /// The actual visitor which can enumerate the subtree nodes in order of creation
    /// </summary>
    public MCTSNodeIteratorInVisitOrder Visitor;


    /// <summary>
    /// Clone method which does a deep copy of the object.
    /// This is needed if we are materializing a tranposition subtree and 
    /// encounter a subnode which is itself tranposition linked; we need to create
    /// another instance of the visitor which can then proceed enumerating indpendently.
    /// </summary>
    /// <returns></returns>
    public object Clone()
    {
      MCTSNodeTranspositionVisitor ret = new MCTSNodeTranspositionVisitor();
      ret.Visitor = Visitor.Clone() as MCTSNodeIteratorInVisitOrder;
      ret.TranspositionRootNWhenVisitsStarted = TranspositionRootNWhenVisitsStarted;
      return ret;
    }
  }
}
