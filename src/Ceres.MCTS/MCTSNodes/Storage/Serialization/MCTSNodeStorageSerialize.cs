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
using System.IO;

using Ceres.Base;
using Ceres.Base.DataType;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCTS.MTCSNodes.Analysis;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Static methods for serializing and deserializing 
  /// the raw MCTS node store.
  /// 
  /// NOTE: Spans do not support indices greater than int.MaxValue, see:
  ///         https://github.com/dotnet/apireviews/tree/main/2016/11-04-SpanOfT
  ///       Since our child arrays can exceed this value, the Save/Restore methods
  ///       below will not work. Instead an NotImplementedException is thrown.
  /// </summary>
  public static class MCTSNodeStorageSerialize
  {
    public static MCTSNodeStore Restore(string directory, string id, bool clearSearchInProgressState = true)
    {
      // Read in miscellaneous information file
      string miscInfoFN = Path.Combine(directory, id + FN_POSTFIX_MISC_INFO);
      MCTSNodeStoreSerializeMiscInfo miscInfo = SysMisc.ReadObj<MCTSNodeStoreSerializeMiscInfo>(miscInfoFN);

      MCTSNodeStore store = new MCTSNodeStore(miscInfo.NumNodesReserved);
      store.Nodes.InsureAllocated(miscInfo.NumNodesAllocated);
      if (miscInfo.RootIndex.Index != 1)
      {
        throw new Exception("Root index must be 1.");
      }
      store.Nodes.Reset(miscInfo.PriorMoves, true);

      long numNodes = SysMisc.ReadFileIntoSpan<MCTSNodeStruct>(Path.Combine(directory, id + FN_POSTFIX_NODES), store.Nodes.Span);
      //store.Nodes.InsureAllocated((int)numNodes);
      store.Nodes.nextFreeIndex = (int)numNodes;

      store.Children.InsureAllocated((int)miscInfo.NumChildrenAllocated);
      long numChildren = SysMisc.ReadFileIntoSpan<MCTSNodeStructChild>(Path.Combine(directory, id + FN_POSTFIX_CHILDREN), store.Children.Span);
      if (numChildren > int.MaxValue)
        throw new NotImplementedException("Implementation restriction: cannot read stores with number of children exceeding int.MaxValue.");
      store.Children.nextFreeBlockIndex = (int)numChildren / MCTSNodeStructChildStorage.NUM_CHILDREN_PER_BLOCK;

      if (clearSearchInProgressState)
      {
        // Reset the search state fields
        MemoryBufferOS<MCTSNodeStruct> nodes = store.Nodes.nodes;
        for (int i = 1; i < store.Nodes.NumTotalNodes; i++)
          nodes[i].ResetSearchInProgressState();
      }

      return store;
    }

    const string FN_POSTFIX_MISC_INFO = "_MCTS_RAW_NODE_MISC_INFO.dat";
    const string FN_POSTFIX_NODES = "_MCTS_RAW_NODE_STORAGE.dat";
    const string FN_POSTFIX_CHILDREN = "_MCTS_RAW_CHILD_STORAGE.dat";

    static string GetPath(string rootPath, string id, string name) => Path.Combine(rootPath, id + name);

    public static void Save(MCTSNodeStore store, string directory, string id)
    {
      if (store.Children.NumAllocatedChildren >= int.MaxValue)
        throw new NotImplementedException("Implementation restriction: cannot write stores with number of children exceeding int.MaxValue.");

      SysMisc.WriteSpanToFile(GetPath(directory, id, FN_POSTFIX_NODES), store.Nodes.Span.Slice(0, store.Nodes.NumTotalNodes));
      SysMisc.WriteSpanToFile(GetPath(directory, id, FN_POSTFIX_CHILDREN), store.Children.Span.Slice(0, (int)store.Children.NumAllocatedChildren));

      MCTSNodeStoreSerializeMiscInfo miscInfo = new MCTSNodeStoreSerializeMiscInfo()
      {
        Description = "",
        PriorMoves = store.Nodes.PriorMoves,
        RootIndex = store.RootIndex,

        NumNodesReserved = store.MaxNodes,
        NumNodesAllocated = store.Nodes.NumTotalNodes,
        NumChildrenAllocated = store.Children.NumAllocatedChildren
      };

      SysMisc.WriteObj(GetPath(directory, id, FN_POSTFIX_MISC_INFO), miscInfo);
    }


  }
}




