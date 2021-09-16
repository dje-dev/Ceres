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
using System;
using System.Diagnostics;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  public partial struct MCTSNodeStruct
  {
    static bool IsValidVSource(in MCTSNodeStruct nodeRef) => !FP16.IsNaN(nodeRef.V)
                                                            && nodeRef.Terminal != Chess.GameResult.NotInitialized;

    /// <summary>
    /// Returns reference to node at specified visitation index at or below specifed node
    /// (or reference to specified nullNodeRef node if does not exist).
    /// </summary>
    /// <param name="node"></param>
    /// <param name="depth"></param>
    /// <param name="nullNodeRef">reference to the null root node</param>
    /// <returns></returns>
    public static ref readonly MCTSNodeStruct SubnodeRefVisitedAtIndex(in MCTSNodeStruct node, int depth,
                                                                       out bool found)
    {
      found = false;
      switch (depth)
      {
        case 0:
          if (IsValidVSource(in node))
          {
            found = true;
            return ref node;
          }
          return ref node;

        case 1:
          if (node.NumChildrenExpanded > 0)
          {
            ref readonly MCTSNodeStruct child = ref node.ChildAtIndexRef(0);
            if (IsValidVSource(in child))
            {
              found = true;
              return ref child;
            }
          }
          return ref node;

        case 2:
          // Only two possibilities to consider: child of child, or sibling (index 1) of child.
          if (node.NumChildrenVisited > 0)
          {
            ref readonly MCTSNodeStruct child = ref node.ChildAtIndexRef(0);

            // Check for a valid child of the child.
            ref readonly MCTSNodeStruct subchildRef = ref node; // dummy initialization
            bool foundSubchild = false;
            if (child.NumChildrenVisited > 0)
            {
              ref readonly MCTSNodeStruct subchild = ref child.ChildAtIndexRef(0);
              if (IsValidVSource(in subchild))
              {
                foundSubchild = true;
                subchildRef = ref subchild;
              }
            }

            // Possibly return sibling if it exists and is valid and better (earlier).
            if (node.NumChildrenVisited > 1)
            {
              ref readonly MCTSNodeStruct siblingRef = ref node.ChildAtIndexRef(1);
              if (IsValidVSource(in siblingRef))
              {
                found = true;
                if (foundSubchild && subchildRef.Index.Index < siblingRef.Index.Index)
                {
                  return ref subchildRef;
                }
                else
                {
                  return ref siblingRef;
                }
              }
            }

            // Return the subchild if it was valid.
            if (foundSubchild)
            {
              found = true;
              return ref subchildRef;
            }
          }

          return ref node;

        default:
          throw new Exception("Internal error: V3 and above not yet supported.");
      }
    }


  }
}
