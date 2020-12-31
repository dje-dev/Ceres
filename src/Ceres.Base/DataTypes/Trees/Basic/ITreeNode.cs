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

using System.Collections.Generic;

#endregion

namespace Ceres.Base.DataType.Trees
{
  /// <summary>
  /// Minimal interface for a node in a tree.
  /// </summary>
	public interface ITreeNode
	{
    /// <summary>
    /// Returns the parent of this node (or default if root)
    /// </summary>
		ITreeNode IParent { get; }

    /// <summary>
    /// Enumerates the children of this node
    /// </summary>
		IEnumerable<ITreeNode> IChildren { get; }

    /// <summary>
    /// Retrieves a single child at a specified index
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
		ITreeNode IChildAtIndex(int index);

    #region Default implementations

    public bool IsRoot => IParent is default(ITreeNode);

    public bool IsLeaf => !IChildren.GetEnumerator().MoveNext();


    // --------------------------------------------------------------------------------------------
    /// <summary>
    /// Returns the depth of this node within the tree (starting at 0 for the root node).
    /// </summary>
    public int IDepth
    {
      get
      {
        ITreeNode node = this;
        int depth = 0;
        while (node.IParent != null)
        {
          depth++;
          node = node.IParent;
        }

        return depth;
      }
    }

    #endregion
  }


}
