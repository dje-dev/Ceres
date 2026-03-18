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

#endregion

namespace Ceres.MCGS.Search.Paths;

/// <summary>
/// Provides an enumerable for traversing an MCGSPath from the leaf node up to the root.
/// </summary>
internal readonly ref struct MCGSPathLeafToRootEnumerable
{
  /// <summary>
  /// The path to enumerate.
  /// </summary>
  public readonly MCGSPath Path;

  /// <summary>
  /// The index of the first local slot to yield.
  /// </summary>
  private readonly int startLocalIndex;


  /// <summary>
  /// Constructor that creates an enumerable that starts with the leaf visit.
  /// </summary>
  public MCGSPathLeafToRootEnumerable(MCGSPath path)
  {
    ArgumentNullException.ThrowIfNull(path);

    Path = path;
    startLocalIndex = path.numSlotsUsed;   // one‑past the leaf index
  }


  /// <summary>
  /// Creates an enumerable that starts at <paramref name="indexLastSlotUsed"/>.
  ///
  /// Enumeration starts at the slot prior to the specified index in the path's local slots,
  /// or the last slot in the parent path if the specified index is -1.
  /// </remarks>
  public MCGSPathLeafToRootEnumerable(MCGSPath path, int indexLastSlotUsed)
  {
    this.Path = path;
    startLocalIndex = indexLastSlotUsed;
  }


  /// <summary>
  /// Returns an enumerator that iterates from the leaf to the root of the path.
  /// </summary>
  /// <returns></returns>
  public MCGSPathVisitEnumerator GetEnumerator() => new(Path, startLocalIndex);
}

/// <summary>
// An efficient (allocation‑free enumerator) that iterates from leaf to root.
/// </summary>
internal ref struct MCGSPathVisitEnumerator
{
private MCGSPath path;
private int currentLocalIndex;
private int currentPathIndex;


/// <summary>
/// Constructor that initializes the enumerator to start at the leaf visit.
/// </summary>
/// <param name="startingPath"></param>
public MCGSPathVisitEnumerator(MCGSPath startingPath) : this(startingPath, startingPath?.numSlotsUsed ?? 0)
{
}


/// <summary>
/// Constructor to start from a specific local index in the path.
/// </summary>
/// <param name="startingPath"></param>
/// <param name="startLocalIndex"></param>
public MCGSPathVisitEnumerator(MCGSPath startingPath, int startLocalIndex)
{
  if (startLocalIndex == -1)
  {
    path = startingPath.parent;
    currentLocalIndex = startingPath.parentSlotIndexLastUsedThisSegment + 1; // one‑past
    currentPathIndex = startingPath.NumVisitsInPath - startingPath.numSlotsUsed; // one‑past composite index
  }
  else
  {
    path = startingPath; // one‑past the desired first slot
    currentLocalIndex = startLocalIndex;
    currentPathIndex = startingPath.NumVisitsInPath - (startingPath.numSlotsUsed - startLocalIndex); // one‑past composite index
  }
}


/// <summary>
/// Moves to the next visit in the path.
/// </summary>
/// <returns></returns>
public bool MoveNext()
{
  if (path == null)
  {
    return false;
  }

  currentLocalIndex--;
  currentPathIndex--;

  if (currentLocalIndex >= 0)
  {
    return true;
  }

  if (path.parent != null)
  {
    currentLocalIndex = path.parentSlotIndexLastUsedThisSegment; // last slot in parent
    path = path.parent;
    return true;
  }

  path = null;
  return false;
}


/// <summary>
/// Returns the current visit in the path.
/// </summary>
public readonly MCGSPathVisitMember Current
{
  get
  {
    if (path == null)
    {
      throw new InvalidOperationException("Enumeration has finished.");
    }
    return new MCGSPathVisitMember(path, currentLocalIndex, currentPathIndex);
  }
}
}


/// <summary>
/// An encapsulated visit member of a path, which includes the path reference,
/// and the local slot index and the composite index of the visit.
/// </summary>
internal readonly record struct MCGSPathVisitMember
{
  /// <summary>
  /// The associated parent MCGSPath.
  /// </summary>
  private readonly MCGSPath path;

  /// <summary>
  /// Gets the zero-based index of the local variable slot associated with this instance.
  /// </summary>
  internal int LocalSlotIndex { get; }

  /// <summary>
  /// Gets the zero-based index of the current path segment within the collection.
  /// </summary>
  internal int PathIndex { get; }

  /// <summary>
  /// Returns if is the root path member.
  /// </summary>
  internal bool IsRoot => PathIndex == 0;

  /// <summary>
  /// Returns if is the leaf path member.
  /// </summary>
  internal bool IsPathLeaf => LocalSlotIndex == path.numSlotsUsed - 1;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="path"></param>
  /// <param name="localIndex"></param>
  /// <param name="compositeIndex"></param>
  internal MCGSPathVisitMember(MCGSPath path, int localIndex, int compositeIndex)
  {
    this.path = path;
    LocalSlotIndex = localIndex;
    PathIndex = compositeIndex;
  }


  /// <summary>
  /// Returns a reference to the path visit at the current slot index.
  /// </summary>
  internal ref MCGSPathVisit PathVisitRef
  {
    [DebuggerStepThrough]
    get
    {
      return ref path.slots[LocalSlotIndex];
    }
  }
}
