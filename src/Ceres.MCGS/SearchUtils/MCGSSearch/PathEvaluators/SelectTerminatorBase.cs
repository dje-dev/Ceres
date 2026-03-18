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
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.PathEvaluators;

/// <summary>
/// Abstract base class for objects that can evaluate nodes, i.e.
/// compute value and policy outputs for a given encoded position.
/// </summary>
public abstract class SelectTerminatorBase
{
  public enum SelectTerminatorMode
  {
    /// <summary>
    /// Normal mode where TryEvaluator returns 
    /// complete result as a LeafEvaluationResult if found.
    /// </summary>
    ReturnEvaluationResult,

    /// <summary>
    /// Alternate mode where default is always returned by TryEvaluate,
    /// but the graph node.
    /// </summary>
    SetAuxilliaryEval
  }


  /// <summary>
  /// Determines mode of return values.
  /// </summary>
  public SelectTerminatorMode Mode { internal set; get; } = SelectTerminatorMode.ReturnEvaluationResult;


  /// <summary>
  /// If this terminator might possibly terminate on an evaluated node (N > 0)
  /// and therefore should be called on inner nodes.
  /// 
  /// This is typically false, except in scenarios such as:
  ///   - node is already terminal
  ///   - node is transposition node with N already greater or equal to new visit N on this path
  /// </summary>
  public virtual bool CanTerminateOnEvaluatedNodes { init; get; } = false;


  /// <summary>
  /// Attempts to evaluate node immediately and returns if successful (else default).
  /// </summary>
  /// <param name="path"></param>
  protected abstract bool DoTryTerminate(MCGSPath path, ref SelectTerminationInfo terminationInfo);


  /// <summary>
  /// BatchPostprocess is called at the end of gathering a batch of leaf nodes,
  /// allowing the terminator to potentially perform postprocessing operation.
  /// 
  /// It is guaranteed that no other operations are concurrently active at this time.
  /// 
  /// TODO: currently neither called nor implemented anywhere.
  /// </summary>
  public virtual void BatchPostprocess()
  {
  }

  
  /// <summary>
  /// Attempts to evaluate node immediately and returns if successful (else default).
  /// </summary>
  /// <param name="path"></param>
  public bool TryEvaluate(MCGSPath path, ref SelectTerminationInfo terminationInfo)
  {
    bool terminated = DoTryTerminate(path, ref terminationInfo);
    if (!terminated)
    {
      return false;
    }
    else if (Mode == SelectTerminatorMode.ReturnEvaluationResult)
    {
      return true;
    }
    else if (Mode == SelectTerminatorMode.SetAuxilliaryEval)
    {
      throw new NotImplementedException();
      //path.LastNodeRef.Node.Annotation.EvalResultAuxilliary = (FP16)result.V;
    }
    else
    {
      throw new NotImplementedException("Unexpected SelectEvaluatorMode");
    }
    
  }

}
