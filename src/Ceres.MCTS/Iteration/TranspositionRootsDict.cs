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
using System.Diagnostics;
using Ceres.Base.Math;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Data structure holding a mapping between hash keys 
  /// of nodes in a search tree and the index of the associated node.
  /// 
  /// Fortunately all updates to this table can be held until
  /// a full batch is gathered (perhaps in parallel) 
  /// and then the bulk update can be applied.
  /// 
  /// Therefore concurrent updates are not expected or supported 
  /// and we can use simple Dictionary implementation.
  /// </summary>
  public class TranspositionRootsDict
  {
    /// <summary>
    /// The number of transposition roots in a tree 
    /// will be much less than the total number of nodes due to transpositions.
    /// </summary>
    const float ESTIMATED_FRACTION_ROOTS = 0.50f;

    /// <summary>
    /// Underlying mapping dictionary.
    /// </summary>
    Dictionary<ulong, int> table;

    /// <summary>
    /// Constructor for a dictionary of specified approximate final size.
    /// Limit max hint to 300mm in case very optimistic value was passed in
    /// (e.g. in case of a search that was requested to be infinite).
    /// </summary>
    /// <param name="sizingHint"></param>
    public TranspositionRootsDict(int estimatedTreeSize) => table = new((int)StatUtils.Bounded((int)(estimatedTreeSize * ESTIMATED_FRACTION_ROOTS), 1000, 300_000_000));


    /// <summary>
    /// Attempts to add entry to index.
    /// </summary>
    /// <param name="hashKey"></param>
    /// <param name="nodeIndex"></param>
    public bool TryAdd(ulong hashKey, int nodeIndexNew, int nodeIndexCurrent, Span<MCTSNodeStruct> nodes)
    {
      Debug.Assert(nodeIndexNew != 1); // Root not allowed to be transposition root

      bool added = table.TryAdd(hashKey, nodeIndexNew);
      if (added)
      {
        nodes[nodeIndexCurrent].IsTranspositionRoot = true;
      }
      return added;
    }

    /// <summary>
    /// Returns underlying IDictionary interface.
    /// </summary>
    public IDictionary<ulong, int> Dictionary => table;

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

    #region Helper methods


    /// <summary>
    /// Updates the table to use an alternate root node 
    /// if the N of the alternate root node exceeds current root's N.
    /// </summary>
    /// <param name="nodes"></param>
    /// <param name="hashKey"></param>
    /// <param name="possibleNewValue"></param>
    /// <param name="possibleNewValueN"></param>
    /// <returns></returns>
    public bool PossiblyUpdateIfNBigger(Span<MCTSNodeStruct> nodes, ulong hashKey, int possibleNewValue, int possibleNewValueN)
    {
      // TODO: much more efficient version for .NET 6 and above
#if NOT
      ref int refValue = ref CollectionsMarshal.GetValueRefOrAddDefault(table, hashKey, out bool exists);
      if (!exists || shouldUpdatePredicate(refValue))
      {
        refValue = possibleNewValue;
        nodes[existingNodeIndex].IsTranspositionRoot = false;
        nodes[possibleNewValue].IsTranspositionRoot = true;
        return true;
      }
#endif

      if (table.TryGetValue(hashKey, out int existingNodeIndex))
      {
        int existingN = nodes[existingNodeIndex].N;
        if (possibleNewValueN > existingN)
        {
          table[hashKey] = possibleNewValue;
          nodes[existingNodeIndex].IsTranspositionRoot = false;
          nodes[possibleNewValue].IsTranspositionRoot = true;
          return true;
        }
      }

      return false;
    }

#if NOT
    /// <summary>
    /// 
    /// </summary>
    /// <param name="hashKey"></param>
    /// <param name="possibleNewValue"></param>
    /// <param name="shouldUpdatePredicate"></param>
    public void AddOrPossiblyUpdate(ulong hashKey, int possibleNewValue, Predicate<int> shouldUpdatePredicate)
    {
      //      #if NET50
      if (table.TryGetValue(hashKey, out int existingValue))
      {
        if (shouldUpdatePredicate(existingValue))
        {
          table[hashKey] = possibleNewValue;
        }
      }
      else
      {
        bool added = table.TryAdd(hashKey, possibleNewValue);
        if (added)
        {
          nodeRef.IsTranspositionRoot = true;
        }
      }

    }
#endif

    /// <summary>
    /// Tries to remove specified entry from dictionary.
    /// </summary>
    /// <param name="hashKey"></param>
    /// <param name="index"></param>
    /// <returns></returns>
    public bool Remove(ulong hashKey, int index)
    {
      // Note that we don't bother resetting the IsTranspositionRoot bit in the node.
      return (table as IDictionary<ulong, int>).Remove(new KeyValuePair<ulong, int>(hashKey, index));
    }

    #endregion
  }

}
