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


#endregion

namespace Ceres.MCTS.Params
{
  public enum TranspositionMode 
  { 
    /// <summary>
    /// No transpositions are tracked or used.
    /// In this case, the position cache should generally be enabled.
    /// </summary>
    None, 

    /// <summary>
    /// A table of transposition roots is maintained.
    /// New nodes are checked against this table,
    /// and if a match is found the position evaluation and policy 
    /// is extracted from the root node already in the tree and copied to the new node.
    /// </summary>
    SingleNodeCopy, 

    /// <summary>
    /// Same behavior as SingleNodeCopy, except that the new node
    /// initially has only the evaluation (not policy) copied from the transposition root.
    /// 
    /// The linked children are created and initialized 
    /// only if the node is visited a second time as part of a subsequent batch.
    /// 
    /// This saves both computational expense and memory (typically about 10% less children allocated).
    /// </summary>
    SingleNodeDeferredCopy, 
    
    /// <summary>
    /// Experimental sharing of single longest subtree
    /// </summary>
    SharedSubtree,

    /// <summary>
    /// Experimental (not yet fully working) 
    /// mode that attaches to transposition root and does not replicate
    /// the subtree, but instead sequentially visits all the nodes in the 
    /// transposition root's subtree.
    /// 
    /// This mode should possibly be abandoned because:
    ///   - the software complexity is high
    ///   - related, the code is still buggy
    ///   - the above mode of SingleNodeDeferredCopy already captures much of the benefits of transpositions
    ///   - tricky issues would have to be dealt with such as multiple visits
    /// </summary>
    MultiNodeBuffered  
  };
}
