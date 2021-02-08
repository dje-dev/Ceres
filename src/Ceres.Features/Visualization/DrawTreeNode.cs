#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System.Collections.Generic;
using System;
using System.Linq;
using System.Drawing;
using System.Diagnostics;
using System.Globalization;
using Ceres.MCTS.MTCSNodes.Struct;

namespace Ceres.Features.Visualization.TreePlot
{
  /// <summary>
  /// Search tree node with (x,y)-coordinates and means to calculate these.
  /// The layout is calculated using Buckheim's algorithm:
  /// C. Buchheim, M. J Unger, and S. Leipert. Improving Walker's algorithm to run in linear time. In Proc. Graph Drawing (GD), 2002. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8757
  /// Implementation here follows closely Bill Mill's python implementation https://github.com/llimllib/pymag-trees/blob/master/buchheim.py
  /// </summary>
  public class DrawTreeNode
  {
    public float X;
    public float Y;
    public DrawTreeNode Parent;
    public List<DrawTreeNode> Children;
    // -1 for root, 0 for nodes of leftmost subtree, 1 for next and so on. Used to determine node coloring.
    public int BranchIndex;
    internal float mod;
    internal float change;
    internal float shift;
    internal DrawTreeNode ancestor;
    internal DrawTreeNode lmostSibling;
    // link between nodes in tree contour if no parent-child relation exists.
    internal DrawTreeNode thread;
    internal int siblingIndex;
    internal int id;

    internal DrawTreeNode(DrawTreeNode parent, MCTSNodeStruct node, int depth, int siblingIndex, ref int identifier)
    {
      X = -1.0f;
      Y = (float)depth;
      id = identifier;
      BranchIndex = parent is null ? -1 : parent.BranchIndex != -1 ? parent.BranchIndex : siblingIndex;
      identifier++;
      Children = new List<DrawTreeNode>();
      int childIndex = 0;
      // Sort children based on N so that heaviest subtree is always drawn leftmost.
      foreach (MCTSNodeStruct child in (from ind in Enumerable.Range(0, node.NumChildrenExpanded) select node.ChildAtIndexRef(ind)).OrderBy(c => -c.N))
      {
        Children.Add(new DrawTreeNode(this, child, depth + 1, childIndex, ref identifier));
        childIndex++;
      }

      Parent = parent;
      mod = 0.0f;
      change = 0.0f;
      shift = 0.0f;
      ancestor = this;
      lmostSibling = null;
      this.siblingIndex = siblingIndex;
      thread = null;
    }

    internal DrawTreeNode Left()
    {
      return thread ?? (Children.Count > 0 ? Children[0] : null);
    }

    internal DrawTreeNode Right()
    {
      return thread ?? (Children.Count > 0 ? Children[^1] : null);
    }

    internal DrawTreeNode LeftSibling()
    {
      return siblingIndex == 0 ? null : Parent.Children[siblingIndex - 1];
    }

    internal DrawTreeNode LeftmostSibling()
    {
      if (lmostSibling is null && siblingIndex != 0)
      {
        lmostSibling = Parent.Children[0];
      }
      return lmostSibling;
    }

    /// <summary>
    /// Calculate tree layout.
    /// </summary>
    internal static (DrawTreeNode, DrawTreeInfo) Layout(MCTSNodeStruct root)
    {
      DrawTreeInfo treeInfo = new DrawTreeInfo();
      int id = 0;
      DrawTreeNode drawRoot = new DrawTreeNode(null, root, 0, 0, ref id);
      drawRoot.FirstWalk();
      float min = drawRoot.SecondWalk(0.0f, float.MaxValue);
      float maxX = float.MinValue;
      float maxDepth = float.MinValue;
      List<int> nodesPerDepth = new List<int>();
      // Shift whole tree so that min x is 0.
      drawRoot.ThirdWalk(-min, treeInfo);
      return (drawRoot, treeInfo);
    }

    /// <summary>
    /// First tree walk (bottom-up) of Buckheim's layout algorithm.
    /// </summary>
    internal void FirstWalk()
    {
      if (Children.Count == 0)
      {
        X = siblingIndex != 0 ? LeftSibling().X + 1.0f : 0.0f;
      }
      else
      {
        DrawTreeNode defaultAncestor = Children[0];
        foreach (DrawTreeNode child in Children)
        {
          child.FirstWalk();
          defaultAncestor = child.Apportion(defaultAncestor);
        }

        ExecuteShifts();
        float midPoint = (Children[0].X + Children[^1].X) / 2;
        DrawTreeNode leftBro = LeftSibling();
        if (leftBro is null)
        {
          X = midPoint;
        }
        else
        {
          X = leftBro.X + 1.0f;
          mod = X - midPoint;
        }
      }
    }

    internal DrawTreeNode Apportion(DrawTreeNode defaultAncestor)
    {
      DrawTreeNode leftSibling = LeftSibling();
      if (!(leftSibling is null))
      {
        DrawTreeNode nodeInnerRight = this;
        DrawTreeNode nodeOuterRight = this;
        DrawTreeNode nodeInnerLeft = leftSibling;
        DrawTreeNode nodeOuterLeft = LeftmostSibling();
        float shiftInnerRight = nodeInnerRight.mod;
        float shiftOuterRight = nodeOuterRight.mod;
        float shiftInnerLeft = nodeInnerLeft.mod;
        float shiftOuterLeft = nodeOuterLeft.mod;
        float shiftLocal;
        while (!(nodeInnerLeft.Right() is null) && !(nodeInnerRight.Left() is null))
        {
          nodeInnerLeft = nodeInnerLeft.Right();
          nodeInnerRight = nodeInnerRight.Left();
          nodeOuterLeft = nodeOuterLeft.Left();
          nodeOuterRight = nodeOuterRight.Right();
          nodeOuterRight.ancestor = this;
          shiftLocal = (nodeInnerLeft.X + shiftInnerLeft) - (nodeInnerRight.X + shiftInnerRight) + 1.0f;
          if (shiftLocal > 0)
          {
            MoveSubtrees(PickAncestor(nodeInnerLeft, defaultAncestor), shiftLocal);
            shiftInnerRight += shiftLocal;
            shiftOuterRight += shiftLocal;
          }
          shiftInnerRight += nodeInnerRight.mod;
          shiftOuterRight += nodeOuterRight.mod;
          shiftInnerLeft += nodeInnerLeft.mod;
          shiftOuterLeft += nodeOuterLeft.mod;
        }
        if (!(nodeInnerLeft.Right() is null) && nodeOuterRight.Right() is null)
        {
          nodeOuterRight.thread = nodeInnerLeft.Right();
          nodeOuterRight.mod += shiftInnerLeft - shiftOuterRight;
        }
        else
        {
          if (!(nodeInnerRight.Left() is null) && nodeOuterLeft.Left() is null)
          {
            nodeOuterLeft.thread = nodeInnerRight.Left();
            nodeOuterLeft.mod += shiftInnerRight - shiftOuterLeft;
          }

          defaultAncestor = this;
        }
      }
      return defaultAncestor;
    }

    internal void ExecuteShifts()
    {
      float cumShift = 0.0f;
      float cumChange = 0.0f;
      DrawTreeNode child;
      for (int i = Children.Count - 1; i >= 0; i--)
      {
        child = Children[i];
        child.X += cumShift;
        child.mod += cumShift;
        cumChange += child.change;
        cumShift += child.shift + cumChange;
      }
    }

    internal DrawTreeNode PickAncestor(DrawTreeNode innerLeftNode, DrawTreeNode defaultAncestor)
    {
      // TODO perhaps there is neater way to check if Ancestor is sibling without relying to the id field.
      // id could then be removed as this is the only place where we use it.
      if (Parent.Children.Any(child => child.id == innerLeftNode.ancestor.id))
      {
        return innerLeftNode.ancestor;
      }
      else
      {
        return defaultAncestor;
      }
    }

    internal void MoveSubtrees(DrawTreeNode nodeLeft, float xShift)
    {
      int subtrees = siblingIndex - nodeLeft.siblingIndex;
      change -= xShift / subtrees;
      shift += xShift;
      nodeLeft.change += xShift / subtrees;
      X += xShift;
      mod += xShift;
    }

    /// <summary>
    /// Second tree walk (top-down) of Buckheim's layout algorithm.
    /// </summary>
    internal float SecondWalk(float xShift, float min)
    {
      X += xShift;

      if (X < min)
      {
        min = X;
      }
      foreach (DrawTreeNode child in Children)
      {
        min = child.SecondWalk(xShift + mod, min);
      }
      return min;
    }

    /// <summary>
    /// Post-processing tree walk to shift whole tree by constant amount horizontally.
    /// Also gather information needed for final plotting.
    /// </summary>
    internal void ThirdWalk(float xShift, DrawTreeInfo treeInfo)
    {
      if (treeInfo.NodesPerDepth.Count <= (int)Y)
      {
        treeInfo.NodesPerDepth.Add(1);
      }
      else
      {
        treeInfo.NodesPerDepth[(int)Y] += 1;
      }
      X += xShift;
      treeInfo.MaxX = Math.Max(X, treeInfo.MaxX);
      treeInfo.MaxDepth = Math.Max(Y, treeInfo.MaxDepth);
      treeInfo.NrNodes++;
      treeInfo.NrLeafNodes += Children.Count > 0 ? 1 : 0;
      foreach (DrawTreeNode child in Children)
      {
        child.ThirdWalk(xShift, treeInfo);
      }
    }
  }
}