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

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Data structure holding a mapping between hash keys 
  /// of nodes in a search tree and the index of the associated node.
  /// </summary>
  public class TranspositionRootsDict
  {
    /// <summary>
    /// Underlying mapping dictionary.
    /// </summary>
    Dictionary<ulong, int> table;

    /// <summary>
    /// Constructor for a dictionary of specified approximate final size.
    /// </summary>
    /// <param name="sizingHint"></param>
    public TranspositionRootsDict(int sizingHint) => table = new Dictionary<ulong, int>(sizingHint);


    /// <summary>
    /// Attempts to add entry to index.
    /// </summary>
    /// <param name="hashKey"></param>
    /// <param name="nodeIndex"></param>
    public void TryAdd(ulong hashKey, int nodeIndex) => table.TryAdd(hashKey, nodeIndex);    

    /// <summary>
    /// Attempts to retrieve the index of the node in the tree having a speciifed hash value.
    /// </summary>
    /// <param name="hashKey"></param>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public bool TryGetValue(ulong hashKey, out int nodeIndex) => table.TryGetValue(hashKey, out nodeIndex);


    /// <summary>
    /// Returns total number of entries currently in the dictionary.
    /// </summary>
    public int Count => table.Count;
  }

}
