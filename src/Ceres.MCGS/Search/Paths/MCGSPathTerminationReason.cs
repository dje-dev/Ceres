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

namespace Ceres.MCGS.Search.Paths;

/// <summary>
/// Represents the reason a MCGS path was terminated.
/// </summary>
public enum MCGSPathTerminationReason : byte
{
  NotYetTerminated,

  /// <summary>
  /// Extended to new leaf node that is to be evaluated by the neural network in this batch.
  /// </summary>
  PendingNeuralNetEval,

  /// <summary>
  /// The neural network evaluation for the new leaf node has already been
  /// completed (either due to prefetch or another overlapping batch).
  /// </summary>
  AlreadyNNEvaluated,

  /// <summary>
  /// Shares (reuses) a neural network evaluation that another path has already requested.
  /// </summary>
  PiggybackPendingNNEval,

  /// <summary>
  /// The move corresponding to the edge leads to a terminal position.
  /// No corresponding node will be created, only the terminal edge.
  /// </summary>
  TerminalEdge,

  /// <summary>
  /// The move corresponding to the edge leads to a terminal position.
  /// A corresponding node will be created.
  /// </summary>
  Terminal,

  /// <summary>
  /// The move triggers a draw by repetition after reaching
  /// an expanded node while in coalesce mode.
  /// </summary>
  DrawByRepetitionInCoalesceMode,

  /// <summary>
  /// In graph mode, an already extant node (transposition) was reached
  /// and already had a sufficient number of visits value can be
  /// directly used without further search.
  /// </summary>
  TranspositionLinkNodeSufficientN,

  /// <summary>
  /// In non-graph mode, an already extant node (transposition) was reached
  /// and its neural network evaluation values will be copied to a new node.
  /// </summary>
  TranspositionCopyValues,

  /// <summary>
  /// Unable to continue backup descent.
  /// </summary>
  Abort,
}
