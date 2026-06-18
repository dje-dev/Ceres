#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System.Collections.Generic;

namespace Ceres.Features.Tournaments.Streaming
{
  /// <summary>
  /// Per-game-thread snapshot buffer maintained by the publisher so that a late-joining
  /// subscriber can be brought fully up to date (current game + all moves played so far)
  /// before receiving live deltas.
  /// </summary>
  internal sealed class ThreadLiveState
  {
    public int ThreadId;

    /// <summary>Last sequence number assigned to an event for this thread.</summary>
    public long Seq;

    /// <summary>The current (or most recent) game's start frame.</summary>
    public GameStartDTO CurrentGame;

    /// <summary>All move frames played in the current game (cleared on each new game).</summary>
    public readonly List<MoveDTO> Moves = new();

    /// <summary>The end frame of the current game once it has finished (else null).</summary>
    public GameEndDTO LastEnd;

    /// <summary>True if the current game has finished.</summary>
    public bool Finished;

    public ThreadLiveState(int threadId) => ThreadId = threadId;
  }
}
