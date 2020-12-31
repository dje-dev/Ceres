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
using System.Collections.Generic;

#endregion

namespace Ceres.Base.DataType.Trees
{
  /// <summary>
  /// Static class of miscellaneous tree utilities (such as traversal).
  /// </summary>
  public static class TreeUtils
  {
    public static IEnumerable<ITreeNode> TraverseBreadthFirst(ITreeNode node)
    {
      yield return node;
      foreach (ITreeNode child in node.IChildren)
        TraverseBreadthFirst(child);
    }


    public static IEnumerable<ITreeNode> TraverseDepthFirst(ITreeNode node)
    {
      foreach (ITreeNode child in node.IChildren)
        TraverseDepthFirst(child);
      yield return node;
    }


    public static List<ITreeNode> Select(ITreeNode node, Predicate<ITreeNode> filter)
    {
      List<ITreeNode> matchingNodes = new List<ITreeNode>();

      foreach (ITreeNode subNode in TraverseBreadthFirst(node))
        if (filter(subNode)) 
          matchingNodes.Add(subNode);

      return matchingNodes;
    }

  }


}
