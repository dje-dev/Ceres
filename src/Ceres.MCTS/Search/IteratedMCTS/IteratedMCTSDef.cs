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

using Ceres.Chess;
using System;


#endregion

namespace Ceres.MCTS.Search.IteratedMCTS
{
  /// <summary>
  /// Defines a iterated MCTS consisting of sequence of search
  /// iterations which expand the search tree and then subsequently
  /// delete or make inactive the incremental subtrees.
  /// </summary>
  [Serializable]
  public class IteratedMCTSDef
  {
    /// <summary>
    /// Type of tree modification operation to apply after each ieteration.
    /// </summary>
    public enum TreeModificationType 
    {
      /// <summary>
      /// Delete the nodes in the tree.
      /// </summary>
      DeleteNodes,

      /// <summary>
      /// Delete the nodes in the tree but transfer the 
      /// position evaluations into the memory cache.
      /// </summary>
      DeleteNodesMoveToCache, 

      /// <summary>
      /// Completely backs out all visits made in the tree
      /// (but does not delete nodes).
      /// </summary>
      ClearNodeVisits
    };

    /// <summary>
    /// Sequence of one or more search iteration to be applied.
    /// </summary>
    public readonly IteratedMCTSStepDef[] StepDefs;


    /// <summary>
    /// The type of modification to be made to
    /// the search tree after each iteration.
    /// </summary>
    public readonly TreeModificationType TreeModification;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="treeModification"></param>
    /// <param name="stepDefs"></param>
    public IteratedMCTSDef(TreeModificationType treeModification, params IteratedMCTSStepDef[] stepDefs)
    {
      TreeModification = treeModification;
      StepDefs = stepDefs;

    }


    /// <summary>
    /// Sets the search limit to be applied to the iterated search.
    /// </summary>
    /// <param name="overallLimit"></param>
    public void SetForSearchLimit(SearchLimit overallLimit)
    {
      foreach (IteratedMCTSStepDef step in StepDefs)
        step.SetForOverallLimit(overallLimit);
    }
  }
}
