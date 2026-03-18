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

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Efficiently serves as a "pointer" to an MCGS node
/// indirect as an integer index into an MCGS array of nodes.
/// 
/// Index zero is reserved to represent a null node.
/// </summary>
[Serializable]
public readonly record struct NodeIndex(int index) : IComparable<NodeIndex>
{
  #region Helpers

  /// <summary>
  /// Instance of the null NodeIndex.
  /// </summary>
  public static readonly NodeIndex Null;


  /// <summary>
  /// Null nodes represent "does not exist"
  /// Importantly the default value of this struct is the null value
  /// </summary>
  public bool IsNull => index == 0;


  /// <summary>
  /// Returns the index of the node in the MCGS node array.
  /// </summary>
  public int Index
  {
    [DebuggerStepThrough]
    get
    {
      return index;
    }
  }

  #endregion

  #region ToString/IComparable

  /// <summary>
  /// Returns a string representation of the node index.
  /// </summary>
  /// <returns></returns>
  public override string ToString() =>  $"<NodeIndex [#{index}]>";
  

  /// <summary>
  /// Compares this node index to another.
  /// </summary>
  /// <param name="other"></param>
  /// <returns></returns>
  public int CompareTo(NodeIndex other) => Index.CompareTo(other.Index);

  #endregion
}
