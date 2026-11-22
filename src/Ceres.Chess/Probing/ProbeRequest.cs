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

using System.Threading;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;

#endregion

namespace Ceres.Chess.Probing;

/// <summary>
/// One asynchronous probe request against a snapshot of a search-graph position.
/// Immutable. The requester identifies the position both by its reconstructed board state
/// (position + history, sufficient for exact repetition/50-move handling in the prober) and
/// by an opaque correlation token it can use to re-find its own bookkeeping on completion.
/// </summary>
public sealed record ProbeRequest
{
  /// <summary>Monotonically increasing request ID (unique within the process).</summary>
  public required long RequestID { get; init; }

  /// <summary>Opaque correlation token round-tripped to the result (not interpreted by the prober).</summary>
  public required object RequesterToken { get; init; }

  /// <summary>Position to search, including prior game/search history
  /// (sufficient for repetition and 50-move handling).</summary>
  public required PositionWithHistory Position { get; init; }

  /// <summary>The move currently dominating MCGS visits at this node (the move under test),
  /// or default if the requester wants an untargeted best-move probe.</summary>
  public MGMove DominantMove { get; init; }

  /// <summary>
  /// When set, the prober searches ONLY this root move (UCI searchmoves semantics) with the full
  /// NodeBudget: the result's BestMove is this move, BestScoreCp its score, PV its line, and
  /// DominantMoveScoreCp / RefutationPV repeat the score and the line after the move so the
  /// requester can consume it as a favourite verification. Default = unrestricted probe.
  /// (The dual-probe advocate's favourite monitor uses this; DominantMove is ignored when set.)
  /// </summary>
  public MGMove RestrictToMove { get; init; }

  /// <summary>
  /// When true the prober additionally delivers an INTERIM ProbeResult (IsInterim = true,
  /// Completed = true) after every completed iteration of its search, through the same callback,
  /// before the usual final result (which is Completed = false when the search was cancelled).
  /// Requesters that set this must expect several callbacks per request; other requesters keep the
  /// exactly-one-callback contract. Used with an effectively unbounded NodeBudget by the continuous
  /// advocate searches, which are ended by their CancellationToken.
  /// </summary>
  public bool ReportInterim { get; init; }

  /// <summary>Approximate search effort in prober nodes.</summary>
  public required int NodeBudget { get; init; }

  /// <summary>Cancellation: the requester cancels all outstanding probes at end of search.</summary>
  public required CancellationToken CancellationToken { get; init; }
}
