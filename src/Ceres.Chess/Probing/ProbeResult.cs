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
using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.Chess.Probing;

/// <summary>
/// Result of one probe. Scores are in centipawns from the side-to-move perspective
/// of the probed position and are used by the requester ONLY for margin/trigger comparisons --
/// never converted into MCGS value space.
/// </summary>
public sealed record ProbeResult
{
  /// <summary>Matches the ProbeRequest.RequestID this result answers.</summary>
  public required long RequestID { get; init; }

  /// <summary>The requester's token from the ProbeRequest, echoed verbatim (correlation).</summary>
  public required object RequesterToken { get; init; }

  /// <summary>If the probe completed normally (false: cancelled, errored, or budget-degenerate;
  /// all other fields except RequestID/RequesterToken are then undefined).</summary>
  public required bool Completed { get; init; }

  /// <summary>True for an interim report (one completed prober iteration) of a request that set
  /// ProbeRequest.ReportInterim; the final result of the request follows later.</summary>
  public bool IsInterim { get; init; }

  /// <summary>Prober's best move from the probed position.</summary>
  public MGMove BestMove { get; init; }

  /// <summary>Score of BestMove (cp, side-to-move perspective).</summary>
  public int BestScoreCp { get; init; }

  /// <summary>Score of the request's DominantMove (cp), when it was supplied and evaluated;
  /// int.MinValue otherwise.</summary>
  public int DominantMoveScoreCp { get; init; } = int.MinValue;

  /// <summary>Principal variation from the probed position, starting with BestMove
  /// (the witness line; length may exceed what the requester grafts).</summary>
  public required IReadOnlyList<MGMove> PV { get; init; }

  /// <summary>When the dominant move was refuted: the refutation line AFTER the dominant move
  /// (i.e. the PV of the reply search), empty otherwise. Grafting a refutation prepends
  /// DominantMove to this line.</summary>
  public IReadOnlyList<MGMove> RefutationPV { get; init; } = Array.Empty<MGMove>();

  /// <summary>Prober iteration depth reached (for requester statistics and trust gating).</summary>
  public int Depth { get; init; }

  /// <summary>
  /// Deepest ply reached on any line (selective depth), or 0 when the source does not report it.
  /// Nominal Depth understates how far a probe actually looked along forcing lines, so this is the
  /// honest measure of how much the probe saw.
  /// </summary>
  public int SelDepth { get; init; }

  /// <summary>
  /// True when the probe stopped because it exhausted NodeBudget rather than finishing its final
  /// iteration. Distinguishes "the budget was the binding constraint" (raise it to get value) from
  /// "the probe converged" (raising it buys nothing) -- which a node count alone cannot tell you.
  /// </summary>
  public bool HitNodeBudget { get; init; }

  /// <summary>Prober node count actually spent (for requester statistics only).</summary>
  public long Nodes { get; init; }

  /// <summary>Wall milliseconds from acceptance of work to completion (for requester statistics only).</summary>
  public double ElapsedMilliseconds { get; init; }
}
