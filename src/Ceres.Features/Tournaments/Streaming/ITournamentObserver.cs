#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.Features.Tournaments.Streaming
{
  /// <summary>
  /// General-purpose observer of live tournament events. This is a transport-agnostic
  /// extension point: any consumer can attach to receive game progress as it happens
  /// (one such consumer is the live network streaming publisher, e.g. used by the
  /// EngineBattle GUI in REMOTE mode, but nothing here is specific to any one client).
  ///
  /// Modelled after the existing TournamentDef.PerGameCallback hook: an optional
  /// [NonSerialized] reference resolved via Def.parentDef and invoked with a simple
  /// null-check. When no observer is attached there is zero behavioral change and
  /// near-zero overhead (a handful of null checks per game/move).
  ///
  /// Implementations MUST be fast, non-blocking and must never throw back into the
  /// tournament worker threads (they are expected to enqueue and return immediately).
  /// </summary>
  public interface ITournamentObserver
  {
    /// <summary>
    /// Invoked once when the tournament starts, before worker threads begin.
    /// </summary>
    void OnTournamentStart(TournamentMetaDTO meta);

    /// <summary>
    /// Invoked when a new game begins on the worker thread with the given stable index.
    /// </summary>
    void OnGameStart(int threadIndex, GameStartDTO game);

    /// <summary>
    /// Invoked after each completed half-move (ply) on the given worker thread.
    /// </summary>
    void OnMove(int threadIndex, MoveDTO move);

    /// <summary>
    /// Invoked when a game completes on the given worker thread (and the global
    /// result has been recorded in the shared statistics).
    /// </summary>
    void OnGameEnd(int threadIndex, GameEndDTO end);

    /// <summary>
    /// Invoked once after all games have completed.
    /// </summary>
    void OnTournamentEnd();

    /// <summary>
    /// Registers a callback that enables any optional, higher-overhead data gathering needed only
    /// while someone is actually watching (e.g. verbose move stats for WDL/top-move data). The
    /// observer invokes it as soon as the first client connects, or immediately if a client is
    /// already connected; if no client ever connects, the callback is never invoked (zero overhead).
    /// </summary>
    void RegisterOnFirstClient(System.Action onFirstClient);
  }
}
