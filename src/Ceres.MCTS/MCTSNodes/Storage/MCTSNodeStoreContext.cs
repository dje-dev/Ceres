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

using Ceres.Base.OperatingSystem;
using Ceres.MCTS.MTCSNodes.Struct;
using System;

#endregion

namespace Ceres.MCTS.MTCSNodes.Storage
{
  /// <summary>
  /// Captures an ambient context within which all access to 
  /// MCTSNodeStores must take place for a given thread.
  /// 
  /// The context is required because node indices are used as "pointers"
  /// within the tree rather than full addresses. Thus the only way 
  /// to tie a given node (at an index) back to a physical context is to
  /// have knowledge of the underlying store to which it belongs.
  /// 
  /// The class SearchContextExecutionBlock class facilitates
  /// the orderly construction and release of these contexts
  /// (in a potentially nested fashion).
  /// </summary>
  public class MCTSNodeStoreContext : IDisposable
  {
    [ThreadStatic]
    static MCTSNodeStore curStore;

    [ThreadStatic]
    static MCTSNodeStoreContext curContext;

    [ThreadStatic]
    static MemoryBufferOS<MCTSNodeStruct> curNodes;

    [ThreadStatic]
    static MemoryBufferOS<MCTSNodeStructChild> curChildren;


    public static MCTSNodeStore Store => curStore;

    public static MemoryBufferOS<MCTSNodeStruct> Nodes => curNodes;

    public static MemoryBufferOS<MCTSNodeStructChild> Children => curChildren;


    public readonly MCTSNodeStore NodeStore;

    public readonly MCTSNodeStoreContext PriorContext;


    public MCTSNodeStoreContext(MCTSNodeStore store)
    {
      NodeStore = store;
      PriorContext = curContext;

      // Update statics
      curStore = store;
      curNodes = store.Nodes.nodes;
      curChildren = store.Children.childIndices;
      curContext = this;
    }

    public void Dispose()
    {
      if (curStore == null) throw new Exception("Internal error: MCTSNodeStoreContext improperly nested");

      // Update statics
      if (PriorContext != null)
      {
        curStore = PriorContext.NodeStore;
        curNodes = PriorContext.NodeStore.Nodes.nodes;
        curChildren = PriorContext.NodeStore.Children.childIndices;
      }
      curContext = PriorContext;
    }
  }
}
