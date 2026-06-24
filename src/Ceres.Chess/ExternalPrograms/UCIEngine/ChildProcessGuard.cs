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
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.External.CEngine
{
  /// <summary>
  /// Tracks child engine processes (e.g. Ceres or Lc0 launched via UCI) and ensures
  /// they are terminated when this (parent) process exits, so that no orphaned engine
  /// processes are left running.
  ///
  /// This is "Layer 1" cleanup: it relies on managed shutdown hooks and therefore covers
  /// the common termination paths -- normal exit, Environment.Exit, an unhandled exception,
  /// Ctrl-C, and POSIX SIGINT/SIGTERM/SIGQUIT (the latter run graceful shutdown on .NET).
  ///
  /// It does NOT cover an unstoppable kill of the parent (SIGKILL / "kill -9", or some
  /// debugger/IDE hard-stops), because in those cases no managed code runs at all. Guarding
  /// against that requires an OS-level mechanism (Windows Job Object / Linux PR_SET_PDEATHSIG)
  /// which is intentionally not implemented here.
  ///
  /// Notes on the APIs used: although additional application domains cannot be created on
  /// modern .NET, AppDomain.CurrentDomain and its ProcessExit / UnhandledException events
  /// remain fully supported on .NET (including .NET 10) and are the documented hooks for
  /// process-lifetime notifications.
  /// </summary>
  public static class ChildProcessGuard
  {
    /// <summary>
    /// Live child processes, keyed by process id. Entries are removed automatically when a
    /// process exits on its own.
    /// </summary>
    static readonly ConcurrentDictionary<int, Process> tracked = new();

    /// <summary>
    /// Holds the POSIX signal registrations so they are not garbage collected (disposal of a
    /// PosixSignalRegistration unregisters its handler).
    /// </summary>
    static readonly List<IDisposable> signalRegistrations = new();

    static ChildProcessGuard()
    {
      // Graceful CLR shutdown (normal return / Environment.Exit), and -- on .NET -- SIGTERM.
      AppDomain.CurrentDomain.ProcessExit += (_, _) => KillAll();

      // Unhandled exception bringing the process down. On modern .NET this cannot prevent
      // termination; we just use it to clean up the children first.
      AppDomain.CurrentDomain.UnhandledException += (_, _) => KillAll();

      // Ctrl-C / Ctrl-Break in a console (cross-platform). Do not set e.Cancel, so the
      // process still terminates after the children are killed.
      Console.CancelKeyPress += (_, _) => KillAll();

      // POSIX signals (Linux/macOS). SIGTERM already triggers graceful shutdown on .NET, but
      // registering explicitly is harmless and also covers SIGINT/SIGQUIT. We do not set
      // Cancel, so default termination still proceeds.
      if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
      {
        foreach (PosixSignal sig in new[] { PosixSignal.SIGINT, PosixSignal.SIGTERM, PosixSignal.SIGQUIT })
        {
          signalRegistrations.Add(PosixSignalRegistration.Create(sig, _ => KillAll()));
        }
      }
    }

    /// <summary>
    /// Registers a freshly started child process to be killed when this process exits.
    /// Safe to call for every launched engine process.
    /// </summary>
    /// <param name="process">a started Process</param>
    public static void Track(Process process)
    {
      if (process == null)
      {
        return;
      }

      tracked[process.Id] = process;

      // Auto-remove from the registry if the engine exits on its own.
      try
      {
        process.EnableRaisingEvents = true;
        process.Exited += (_, _) => tracked.TryRemove(process.Id, out _);
      }
      catch
      {
        // Process may have already exited; ignore.
      }
    }

    /// <summary>
    /// Terminates all tracked child processes (and their descendants). Idempotent and never throws.
    /// </summary>
    static void KillAll()
    {
      foreach (Process process in tracked.Values)
      {
        try
        {
          if (!process.HasExited)
          {
            process.Kill(entireProcessTree: true);
          }
        }
        catch
        {
          // Best effort: process may have already exited or be inaccessible.
        }
      }
    }
  }
}
