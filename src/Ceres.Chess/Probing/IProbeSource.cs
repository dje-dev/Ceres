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

namespace Ceres.Chess.Probing;

/// <summary>
/// Abstraction over an asynchronous exact-search probe service (PICKET's probe farm): an
/// in-process engine pool, an external-process adapter, or a test double. A probe answers
/// "what does a real search say about this position, and about this move in particular",
/// and nothing here presumes how it arrives at the answer.
///
/// Contract:
///  - SubmitProbe never blocks and never throws for capacity reasons; it returns false when the
///    request cannot be accepted (queue full, source unavailable), in which case no result will
///    be delivered for it. The caller treats false as a silent no-op.
///  - Each accepted request produces EXACTLY ONE invocation of the completion callback supplied
///    at session start (including for cancellations, with Completed == false), on an arbitrary
///    thread. The callback must therefore only enqueue; it must never touch the caller's graph.
///  - BeginSession/EndSession bracket one MCGS search (one move). EndSession cancels all
///    outstanding requests of the session; late results after EndSession are dropped by the
///    source (never delivered).
///  - A source may host multiple concurrent sessions (one per concurrent search, as when a
///    tournament plays several games in one process). Each result is routed to the session that
///    submitted it, and ending one session leaves the others untouched.
/// </summary>
public interface IProbeSource
{
  /// <summary>
  /// Begins a probe session for one search, supplying the sink for all of that search's
  /// completions, and returns the handle used to submit its probes. A source may host any
  /// number of concurrent sessions; results are routed to the session that submitted them.
  /// </summary>
  IProbeSession BeginSession(Action<ProbeResult> onProbeCompleted);


  /// <summary>
  /// Human-readable description of the probing engine and its configuration (engine name, network,
  /// worker count, table size). Reported in hybrid statistics dumps so a research log records WHICH
  /// prober produced the numbers -- without it a saved run is not reproducible from its output.
  /// Default implementation keeps existing sources source-compatible.
  /// </summary>
  string Description => GetType().Name;
}


/// <summary>
/// One search's view of a probe source: the unit that owns a completion callback, its own
/// in-flight accounting, and its own cancellation.
///
/// The split between source and session is what allows the expensive, position-independent
/// state (network weights, worker threads, transposition table) to be shared process-wide
/// while everything genuinely per-search stays isolated. Concurrent searches therefore share
/// one registered source rather than needing one source apiece.
///
/// Ending a session is idempotent, and Dispose ends it.
/// </summary>
public interface IProbeSession : IDisposable
{
  /// <summary>Submits one probe. Returns false if not accepted (silent no-op for the caller).</summary>
  bool SubmitProbe(ProbeRequest request);

  /// <summary>Ends the session: cancels its outstanding probes; no further callbacks will occur
  /// for this session after this method returns. Other sessions are unaffected.</summary>
  void EndSession();

  /// <summary>Approximate count of this session's accepted-but-uncompleted probes
  /// (flow control / diagnostics).</summary>
  int NumInFlight { get; }
}
