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

using Ceres.MCTS.MTCSNodes.Storage;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  /// <summary>
  /// Efficiently serves as a "pointer" to an MCTS node
  /// by representing as a 4 byte index into an MCTS storage array
  /// 
  /// </summary>
  [Serializable]
  public readonly struct MCTSNodeStructIndex : IEquatable<MCTSNodeStructIndex>, IComparable<MCTSNodeStructIndex>
  {
    #region Data

    private readonly int index;

    #endregion

    #region Helpers

    public static readonly MCTSNodeStructIndex Null = new MCTSNodeStructIndex();

    /// <summary>
    /// Null nodes represent "does not exist"
    /// Importantly the default value of this struct is the null value
    /// </summary>
    public bool IsNull => index == 0;

    public bool IsRoot => index == 1;

    public int Index
    {
      [DebuggerStepThrough]
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return index;
      }
    }

    #endregion
    #region Constructor/conversion

    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="index"></param>
    public MCTSNodeStructIndex(int index) => this.index = index;

    //    // TODO: probably delete this, probably cleaner to not implement this and instead force be explict construction
    // public static implicit operator MCTSNodeIndex(uint index) => new MCTSNodeIndex(index);

    #endregion

    #region As Refs

    // WARNING: Visual Studio editor crashes if this line is enabled
    //public ref readonly MCTSNode RefReadonly => throw new Exception("Not imlementable, see UCTNode.Index method for info");// ref UCTNodeStorage.nodes[index];

    public ref MCTSNodeStruct Ref
    {
      [DebuggerStepThrough]
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return ref MCTSNodeStoreContext.Nodes[index];
      }
    }

    #endregion


    #region ToString/IEquatable

    public readonly override string ToString()
    {
      return $"<MCTSNodeIndex [#{index}]";
    }

    public readonly bool Equals(MCTSNodeStructIndex other) => index == other.index;

    public readonly override bool Equals(object obj)
    {
      if (obj.GetType() != typeof(MCTSNodeStructIndex))
      {
        return false;
      }

      return ((MCTSNodeStructIndex)obj) == this;
    }

    public readonly override int GetHashCode() => (int)index;

    public int CompareTo(MCTSNodeStructIndex other) => Index.CompareTo(other.Index);

    public static bool operator ==(MCTSNodeStructIndex left, MCTSNodeStructIndex right) => left.Equals(right);

    public static bool operator !=(MCTSNodeStructIndex left, MCTSNodeStructIndex right) => !(left == right);

    #endregion

    #region Module Initiailzation

    [ModuleInitializer]
    internal static void ModuleInitializerCheckSize()
    {
      Debug.Assert(Marshal.SizeOf<MCTSNodeStructIndex>() == 4);
    }
    
    #endregion
  }

}
