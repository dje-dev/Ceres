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


#endregion


namespace Ceres.Base.DataType.Trees
{
  /// <summary>
  /// Enumeration of different methods for traversing all nodes of a tree.
  /// </summary>
  public enum TreeTraversalType 
  { 
    /// <summary>
    /// An unspecified order (typically the most performant)
    /// </summary>
    Unspecified,

    /// <summary>
    /// In order by which the tree nodes were created
    /// </summary>
    Sequential,

    /// <summary>
    /// Children before parent
    /// </summary>
    BreadthFirst, 

    /// <summary>
    /// Parent before children
    /// </summary>
    DepthFirst
  };


}
