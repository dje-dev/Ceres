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

using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.Params;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// Structure that transiently holds the output of a LeafEvaluator
  /// (until the result can be transferred into an MCTS node).
  /// </summary>
  [Serializable]
  [StructLayout(LayoutKind.Sequential, Pack = 2)]
  public struct LeafEvaluationResult
  {
    /// <summary>
    /// Policy win probability percentage.
    /// </summary>
    public FP16 WinP;

    /// <summary>
    /// Policy loss probability percentage.
    /// </summary>
    public FP16 LossP;

    /// <summary>
    /// Moves left value (if any).
    /// </summary>
    public FP16 M;

    /// <summary>
    /// Transiently holds policy array within which the policy resides
    /// (but will be released after the policy is applied by being copied into a search node)
    /// </summary>
    private Memory<CompressedPolicyVector> policyArray;

    /// <summary>
    /// Index in the policyArray of this policy value.
    /// </summary>
    private short policyArrayIndex;

    /// <summary>
    /// Terminal status of node.
    /// </summary>
    public GameResult TerminalStatus { get; set; }

    /// <summary>
    /// Policy draw probability percentage.
    /// </summary>
    public float DrawP
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        float winP = WinP.ToFloatApprox;
        float lossP = LossP.ToFloatApprox;
        float v = winP - lossP;
        return ParamsSelect.VIsForcedResult(v) ? 0 
                                               : 1.0f - (winP - lossP);
#if NOT
        float correct = ParamsSelect.VIsForcedResult(V) ? 0 : 1.0f - (WinP + LossP);
        float tryx;

        if (ParamsSelect.VIsForcedResult(V)) 
          tryx = 0;
        else
          tryx = 1.0f - (WinP.ToFloatApprox + LossP.ToFloatApprox);
        if (MathF.Abs(tryx - correct) > 0.0001)
          throw new Exception($"BAD { tryx } correct { correct} from {WinP} {LossP}  {ToF(WinP)} {ToF(LossP)}");
        return tryx;
#endif
      }
    }

    /// <summary>
    /// Value estimate (logistic).
    /// </summary>
    public float V => WinP.ToFloatApprox - LossP.ToFloatApprox;

    /// <summary>
    /// If the underlying policy has been released.
    /// </summary>
    public bool PolicyIsReleased => policyArrayIndex == -1;

    /// <summary>
    /// If the structure has been initialized.
    /// </summary>
    public bool IsNull => TerminalStatus == GameResult.NotInitialized;


    /// <summary>
    /// In the case of virtual subtrees (linked to another transposition subnode)
    /// leaf gathering descent may encounter a transposition linked node and yield that
    /// as the pseudo-leaf. However potentially more than one nodes was still allocated
    /// in the batch to this subtree. In that case, we probe the linked subtree
    /// multiple times (if possible) and store these extra evaluations in the ExtraResults field
    /// </summary>
    public List<LeafEvaluationResult> ExtraResults;


    /// <summary>
    /// Constructor from specified values (with no policy).
    /// </summary>
    /// <param name="terminalStatus"></param>
    /// <param name="winP"></param>
    /// <param name="lossP"></param>
    /// <param name="m"></param>
    public LeafEvaluationResult(GameResult terminalStatus, FP16 winP, FP16 lossP, FP16 m)
    {
      Debug.Assert(terminalStatus != GameResult.NotInitialized);

      TerminalStatus = terminalStatus;
      WinP = winP;
      LossP = lossP;
      M = m;
      policyArrayIndex = -1;
      policyArray = null;
      ExtraResults = null;
    }

    /// <summary>
    /// Re-initializes from specified values (with no policy).
    /// </summary>
    /// <param name="terminalStatus"></param>
    /// <param name="winP"></param>
    /// <param name="lossP"></param>
    /// <param name="m"></param>
    public void Initialize(GameResult terminalStatus, FP16 winP, FP16 lossP, FP16 m)
    {
      Debug.Assert(terminalStatus != GameResult.NotInitialized);

      TerminalStatus = terminalStatus;
      WinP = winP;
      LossP = lossP;
      M = m;
      policyArrayIndex = -1;
      policyArray = null;
      ExtraResults = null;
    }


    /// <summary>
    /// Constructor from specified values (including policy reference).
    /// </summary>
    /// <param name="terminalStatus"></param>
    /// <param name="winP"></param>
    /// <param name="lossP"></param>
    /// <param name="m"></param>
    /// <param name="policyArray"></param>
    /// <param name="policyArrayIndex"></param>
    public LeafEvaluationResult(GameResult terminalStatus, FP16 winP, FP16 lossP, FP16 m, Memory<CompressedPolicyVector> policyArray, short policyArrayIndex)
    {
      Debug.Assert(terminalStatus != GameResult.NotInitialized);

      TerminalStatus = terminalStatus;
      WinP = winP;
      LossP = lossP;
      M = m;

      this.policyArrayIndex = policyArrayIndex;
      this.policyArray = policyArray;

      ExtraResults  = null;
    }

    /// <summary>
    /// Reference to underlying policy.
    /// </summary>
    public ref readonly CompressedPolicyVector PolicyRef
    {
      get
      {
        if (policyArrayIndex == -1) 
          throw new Exception("Internal error: access to release policy object");
        return ref policyArray.Span[policyArrayIndex];
      }
    }


    /// <summary>
    /// Memory reference to underlying policy.
    /// </summary>
    public (Memory<CompressedPolicyVector> policies, int index) PolicyInArray
    {
      set
      {
        Debug.Assert(policyArrayIndex == -1 && value.index != -1);
        this.policyArray = value.policies;
        this.policyArrayIndex = (short)value.index;
      }
    }


    /// <summary>
    /// Policy as a CompressedPolicyVector.
    /// </summary>
    public CompressedPolicyVector PolicySingle
    {
      set
      {
        Debug.Assert(policyArrayIndex == -1);
        policyArray = new CompressedPolicyVector[1] { value };
        policyArrayIndex = 0;
      }
    }

    /// <summary>
    /// Releases underlying policy value.
    /// </summary>
    public void ReleasePolicyValue()
    {
      policyArray = null;
      policyArrayIndex = -1;
    }

  }

}
