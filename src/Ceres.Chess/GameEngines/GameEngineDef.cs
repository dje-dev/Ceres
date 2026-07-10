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
using Ceres.Chess.NNEvaluators.Defs;

#endregion

namespace Ceres.Chess.GameEngines
{
  /// <summary>
  /// Abstract base class for definitions of game engines,
  /// which could be internal Ceres engine, or external LC0 or UCI engines.
  /// </summary>
  [Serializable]
  public abstract class GameEngineDef
  {
    /// <summary>
    /// Identifying string of the engine.
    /// </summary>
    public readonly string ID;

    /// <summary>
    /// Index of processor group on which engine should execute.
    /// </summary>
    public int ProcessorGroupID;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    public GameEngineDef(string id)
    {
      ID = id;
    }


    /// <summary>
    /// If the NodesPerGame time control mode is supported.
    /// </summary>
    public abstract bool SupportsNodesPerGameMode { get; }


    /// <summary>
    /// Abstract virtual mthod to create underlying engine.
    /// </summary>
    /// <returns></returns>
    public abstract GameEngine CreateEngine();


    /// <summary>
    /// If applicable, sets the device indices of the underlying evaluator to the
    /// specified (absolute) device IDs.
    ///
    /// Supports evaluators spanning one or multiple devices (e.g. a "GPU:0,1" spec); the
    /// number of supplied IDs must match the evaluator's device count. Pooled evaluators
    /// (and engines without an evaluator) are left unchanged, allowing concurrent workers
    /// to each be assigned a distinct set of GPUs. This is the single mechanism for
    /// (re)assigning the devices of an engine's evaluator.
    /// </summary>
    /// <param name="deviceIDs"></param>
    public virtual void TrySetDeviceIndicesIfNotPooled(int[] deviceIDs)
      => GetEvaluatorDef()?.TrySetDeviceIndices(deviceIDs);

    /// <summary>
    /// Whether this engine definition is for an in-process Ceres engine (MCTS or MCGS).
    /// This is the discriminator for Ceres-specific behavior and is independent of
    /// GetEvaluatorDef (e.g. an external LC0 engine has an evaluator but IsCeresEngine is false).
    /// </summary>
    public virtual bool IsCeresEngine => false;

    /// <summary>
    /// The neural network evaluator definition of this engine, or null if it has none.
    /// Returned for any engine that uses an NN evaluator (in-process Ceres as well as
    /// external engines such as LC0 whose device assignment is derived from it), so it is
    /// used for device assignment but is NOT a test for "is a Ceres engine" (use IsCeresEngine).
    /// </summary>
    public virtual NNEvaluatorDef GetEvaluatorDef() => null;
    public virtual void DisableTreeReuse() { }
    public virtual bool GetReusePositionEvaluationsFromOther() => false;
    public virtual void SetReusePositionEvaluationsFromOther(bool value) { }
    public virtual bool GetFutilityPruningStopSearchEnabled() => false;
  }
}
