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

namespace Ceres.MCGS.Search;

/// <summary>
/// Custom enumeration pattern for processing array of children
/// selected for expansion during selection phase.
/// 
/// In MultiPass mode multiple passes are made over the selected children.
/// The children with lowest indices are the "already visited" ones.
/// </summary>
public ref struct SelectVisitsEnumerator
{
  public enum VisitsPhase
  {
    /// <summary>
    /// Operating in single pass mode: simply iterate over all children.
    /// </summary>
    SinglePass,

    /// <summary>
    /// Pass 1: the children not yet visited (high indices) processed once.
    /// </summary>
    MultiPassNotYetVisited,

    /// <summary>
    /// Pass 2: a first pass over only the already visited children.
    /// </summary>
    MultiPassVisitedLaunchParallel,

    /// <summary>
    /// Pass 3: a second pass over the already visited children.
    /// </summary>
    MultiPassVisitedProcessNonParallel
  }

  private readonly int numChildrenAlreadyVisited;
  private readonly int notExpandedCount; // used in multipass mode
  private readonly int totalCount;
  private readonly bool multipass;
  private int current;


  /// <summary>
  /// Consructor.
  /// </summary>
  /// <param name="numChildrenToConsider"></param>
  /// <param name="numChildrenAlreadyVisited"></param>
  /// <param name="multipass"></param>
  public SelectVisitsEnumerator(int numChildrenToConsider, int numChildrenAlreadyVisited, bool multipass)
  {
    this.multipass = multipass;
    this.numChildrenAlreadyVisited = numChildrenAlreadyVisited;

    if (multipass)
    {
      // Compute steps for multipass: "not-expanded" + two passes for already visited.
      notExpandedCount = numChildrenToConsider - numChildrenAlreadyVisited;
      totalCount = notExpandedCount + 2 * numChildrenAlreadyVisited;
    }
    else
    {
      // In single pass mode: iterate over all children.
      // The numChildrenAlreadyVisited is not used and nor is the two-pass scheme.
      notExpandedCount = 0;
      totalCount = numChildrenToConsider;
    }

    current = -1;
  }


  /// <summary>
  /// Returns enumerator over the visits.
  /// </summary>
  /// <returns></returns>
  public readonly SelectVisitsEnumerator GetEnumerator() => this;


  /// <summary>
  /// Advances to next visit.
  /// </summary>
  /// <returns></returns>
  public bool MoveNext() => ++current < totalCount;


  /// <summary>
  /// Returns current visit.
  /// </summary>
  public (VisitsPhase phase, int index) Current
  {
    get
    {
      if (!multipass)
      {
        // SinglePass mode: simply iterate over all children.
        return (VisitsPhase.SinglePass, current);
      }
      else
      {
        // Multipass mode
        if (current < notExpandedCount)
        {
          // First segment: NotYetVisited phase.
          return (VisitsPhase.MultiPassNotYetVisited, current + numChildrenAlreadyVisited);
        }
        else if (current < notExpandedCount + numChildrenAlreadyVisited)
        {
          // Second segment: first pass over visited elements.
          return (VisitsPhase.MultiPassVisitedLaunchParallel, current - notExpandedCount);
        }
        else
        {
          // Third segment: second pass over visited elements.
          return (VisitsPhase.MultiPassVisitedProcessNonParallel, current - notExpandedCount - numChildrenAlreadyVisited);
        }
      }
    }
  }
}
