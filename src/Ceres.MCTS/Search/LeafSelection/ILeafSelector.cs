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

using Ceres.Base.DataTypes;
using Ceres.MCTS.MTCSNodes;

#endregion

namespace Ceres.MCTS.Search
{
  /// <summary>
  /// Interface implemented by all classes that implement 
  /// leaf selection algorithms, i.e. determine the next best leaf 
  /// (or set of leafs) to be expanded in the tree.
  /// </summary>
  public interface ILeafSelector
  {
    /// <summary>
    /// Maximum number of conccurrent selectors supported.
    /// Selectors typically vary by using slightly different parameters
    /// in the leaf selection process (to reduce collisions).
    /// </summary>
    public const int MAX_SELECTORS = 2;


    /// <summary>
    /// The index of this selector (if multiple selectors can be concurrently active).
    /// </summary>
    int SelectorID { get;}


    /// <summary>
    /// Worker method that selects a next batch (of specified size) if 
    /// leaf nodes to be expanded.
    /// </summary>
    /// <param name="root"></param>
    /// <param name="numVisitsTargeted"></param>
    /// <param name="vLossDynamicBoost"></param>
    /// <returns></returns>
    ListBounded<MCTSNode> SelectNewLeafBatchlet(MCTSNode root, int numVisitsTargeted, float vLossDynamicBoost);


    /// <summary>
    /// Insures that a specified node has been annotated.
    /// </summary>
    /// <param name="node"></param>
    void InsureAnnotated(MCTSNode node);


    /// <summary>
    /// Shuts down the selector.
    /// </summary>
    void Shutdown();
  }
}


