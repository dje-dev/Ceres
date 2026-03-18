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
using System.Diagnostics;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// RAII-style lock guard specifically for GNode instances using the Dispose pattern.
/// </summary>
public readonly ref struct NodeLockBlock : IDisposable
{
  /// <summary>
  /// Gets the associated node for this instance.
  /// </summary>
  public readonly GNode Node;


  /// <summary>
  /// Acquires the lock for the specified node.
  /// </summary>
  /// <param name="node">The node to lock</param>
  public NodeLockBlock(GNode node)
  {
    Debug.Assert(!node.IsNull);

    Node = node;
    node.AcquireLock();
  }


  /// <summary>
  /// Acquires the lock for the specified (possibly null) node.
  /// </summary>
  /// <param name="node">The node to lock</param>
  /// <param name="allowNullNode">If true, a null node is acceptable (making this a no-op); otherwise asserts.</param>
  public NodeLockBlock(GNode node, bool allowNullNode)
  {
    Node = node;

    if (node.IsNull)
    {
      Debug.Assert(allowNullNode, "Node cannot be null when allowNullNode is false.");
    }
    else
    {
      node.AcquireLock();
    }
  }


  /// <summary>
  /// Releases the lock when the block exits scope.
  /// </summary>
  public void Dispose()
  {
    if (!Node.IsNull)
    {
      Node.ReleaseLock();
    }
  }
}
