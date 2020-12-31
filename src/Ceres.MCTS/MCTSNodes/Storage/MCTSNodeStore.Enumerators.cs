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

using System.Runtime.CompilerServices;
using Ceres.Base.OperatingSystem;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.MTCSNodes.Storage
{
  public partial class MCTSNodeStore : IDisposable
  {
    /// <summary>
    /// Methods at MCTSNodeStore related to enumeration of children.
    /// 
    /// NOTE: this is not currently used but possibly could/should be used.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <param name="overrideMaxIndex"></param>
    /// <returns></returns>
    public ChildEnumeratorImpl ChildrenExpandedEnumerator(MCTSNodeStructIndex nodeIndex, int overrideMaxIndex = int.MaxValue)
      => new ChildEnumeratorImpl(this, nodeIndex, overrideMaxIndex);


    /// <summary>
    /// Structure used to facilitate enumeration over children.
    /// </summary>
    [SkipLocalsInit]
    public unsafe ref struct ChildEnumeratorImpl
    {
      MCTSNodeStore Store; 
      MCTSNodeStructIndex NodeIndex;
      int MaxIndex;

      public ChildEnumeratorImpl(MCTSNodeStore store, MCTSNodeStructIndex nodeIndex, int overrideMaxIndex = int.MaxValue)
      {
        Store = store;
        NodeIndex = nodeIndex;
        MaxIndex = Math.Min(overrideMaxIndex, Store.Nodes.nodes[nodeIndex.Index].NumChildrenExpanded - 1) ;
      }

      public ChildEnumeratorImplState GetEnumerator() => new ChildEnumeratorImplState(Store, NodeIndex, MaxIndex);

      public unsafe ref struct ChildEnumeratorImplState
      {
        // TODO: according to #lowlevel discussion, "only structs with <= 4 fields can be promoted"
        //       also "struct fields must (generally) be simple types and not other structs"
        //       So we need to compress this down somehow.
        //       See dotnet/runtime #37924
        MCTSNodeStore Store;
        MCTSNodeStruct* nodes;
        readonly Span<MCTSNodeStructChild> childSpan;
        uint index;
        readonly int endIndex;

        public ref MCTSNodeStruct Current
        {
          [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
          get
          {
            Debug.Assert(index <= endIndex);

            if (false && index < endIndex - 1) // prefetching may be worse performance
            {
              MCTSNodeStructIndex childIndex = childSpan[(int)index + 1].ChildIndex;
              void* nodePtr = Unsafe.AsPointer<MCTSNodeStruct>(ref nodes[childIndex.Index]);
              System.Runtime.Intrinsics.X86.Sse.Prefetch1(nodePtr);
            }
            return ref nodes[childSpan[(int)index].ChildIndex.Index];
          }
        }

        internal ChildEnumeratorImplState(MCTSNodeStore store, MCTSNodeStructIndex nodeIndex, int maxIndex)
        {
          Store = store;
          childSpan = store.Children.SpanForNode(nodeIndex);
          index = UInt32.MaxValue;
          endIndex = (int)maxIndex;
          nodes = (MCTSNodeStruct*)Store.Nodes.nodes.RawMemory;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool MoveNext() => unchecked(++index) <= endIndex;

        private static void ThrowInvalidOp() => throw new InvalidOperationException();
      }
    }

  }

}
