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

#endregion

namespace Ceres.MCGS.Graphs.Enumerators;

public readonly ref struct GEdgeWithIndexEnumerable
{
  public readonly GNode Node;

  public GEdgeWithIndexEnumerable(GNode node) => Node = node;

  public readonly GEdgeWithIndexEnumerator GetEnumerator() => new(Node);
}
