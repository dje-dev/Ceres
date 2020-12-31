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

#endregion

namespace Ceres.Base.Threading
{
  /// <summary>
  /// Disposable structure linked to a specified LockSet 
  /// which can be used in a using statement to:
  ///   - acquires lock of item with specified index upon construction, and
  ///   - release lock of item upon exit of using block (when Dispose is called)
  /// </summary>
  public struct LockSetBlock : IDisposable
  {
    public readonly LockSet Set;
    public readonly int ItemIndex;

    /// <summary>
    /// Constructor over a specified LockSet to lock item with a specified index.
    /// </summary>
    /// <param name="set"></param>
    /// <param name="itemIndex"></param>
    public LockSetBlock(LockSet set, int itemIndex)
    {
      Set = set;
      ItemIndex = itemIndex;

      set.Acquire(itemIndex);
    }


    /// <summary>
    /// Dispose method which is called upon exit of using statement.
    /// </summary>
    public void Dispose() => Set.Release(ItemIndex);

  }

}
