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

using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GParents;
using System;

#endregion

namespace Ceres.MCGS.Graphs.Enumerators;

/// <summary>
/// Serves as an allocation-free enumerable over the parent indices of a specified node.
/// </summary>
internal ref struct ParentIndicesEnumerable
{
  /// <summary>
  /// Index of the node over which we are enumerating parents.
  /// </summary>
  public NodeIndex NodeIndex;

  /// <summary>
  /// Underlying graph store containing this node.
  /// </summary>
  public GParentsStore Store;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="store"></param>
  /// <param name="childIndex"></param>
  public ParentIndicesEnumerable(GParentsStore store, NodeIndex childIndex)
  {
    NodeIndex = childIndex;
    Store = store ?? throw new ArgumentNullException(nameof(store));
  }


  /// <summary>
  /// Returns an enumerator that iterates through the parent indices of the specified node.
  /// </summary>
  /// <returns></returns>
  public ParentIndexEnumerator GetEnumerator()
  {
    return new ParentIndexEnumerator(Store, NodeIndex);
  }
}
