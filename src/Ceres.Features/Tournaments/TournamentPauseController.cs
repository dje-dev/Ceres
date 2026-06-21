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
using System.Threading;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Coordinates a cooperative pause/resume of all worker threads in a running tournament
  /// (driven by the Ctrl-P console command). Worker threads call WaitIfPaused at an
  /// end-of-game boundary; when a pause is requested they park there until resumed.
  ///
  /// Membership is dynamic: each worker registers when it enters its game loop and
  /// deregisters when it exits, so threads that finish all of their games (and leave) do
  /// not stall the "all threads parked" condition the key-monitor waits on.
  /// </summary>
  public sealed class TournamentPauseController
  {
    readonly object sync = new();

    bool pauseRequested;

    /// <summary>Number of worker threads currently inside their game loop.</summary>
    int numActive;

    /// <summary>Number of worker threads currently parked at the pause gate.</summary>
    int numWaiting;


    /// <summary>
    /// Registers the calling thread as an active participant (call when entering the game loop).
    /// </summary>
    public void RegisterActive()
    {
      lock (sync)
      {
        numActive++;
      }
    }


    /// <summary>
    /// Deregisters the calling thread (call when leaving the game loop, e.g. in a finally).
    /// Wakes the key-monitor so it can recompute whether all remaining threads are parked.
    /// </summary>
    public void DeregisterActive()
    {
      lock (sync)
      {
        numActive--;
        Monitor.PulseAll(sync);
      }
    }


    /// <summary>
    /// End-of-game checkpoint called by each worker thread. If a pause is in effect the thread
    /// parks here until a subsequent Resume; otherwise it returns immediately (cheap, no-op).
    /// </summary>
    public void WaitIfPaused()
    {
      lock (sync)
      {
        if (!pauseRequested)
        {
          return;
        }

        numWaiting++;
        Monitor.PulseAll(sync);          // let the key-monitor observe that another thread parked
        while (pauseRequested)
        {
          Monitor.Wait(sync);
        }
        numWaiting--;
      }
    }


    /// <summary>
    /// True if a pause is currently requested (i.e. between the first Ctrl-P and the resuming one).
    /// </summary>
    public bool IsPauseRequested
    {
      get { lock (sync) return pauseRequested; }
    }


    /// <summary>
    /// Requests a pause and blocks until every active worker thread has parked at its end-of-game
    /// checkpoint (or until shouldAbort returns true, e.g. the tournament is shutting down).
    /// Intended to be called from the key-monitor thread on the first Ctrl-P.
    /// Returns true if all threads parked, or false if aborted before that happened.
    /// </summary>
    /// <param name="shouldAbort">Predicate polled while waiting; returning true abandons the wait.</param>
    public bool RequestPauseAndWaitAllParked(Func<bool> shouldAbort)
    {
      lock (sync)
      {
        pauseRequested = true;

        // Wait until all currently-active threads are parked. numActive can drop as threads finish
        // their games and deregister; once every remaining active thread is parked we are done.
        while ((numActive == 0 || numWaiting < numActive) && !(shouldAbort?.Invoke() ?? false))
        {
          Monitor.Wait(sync, 200);
        }

        return !(shouldAbort?.Invoke() ?? false);
      }
    }


    /// <summary>
    /// Releases all parked worker threads and clears the pause (the resuming Ctrl-P).
    /// </summary>
    public void Resume()
    {
      lock (sync)
      {
        pauseRequested = false;
        Monitor.PulseAll(sync);
      }
    }
  }
}
