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
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

#endregion

namespace Ceres.Chess.NetEvaluation.Batch
{
  /// <summary>
  /// Represents the a evaluation from a neural network
  /// of a single position.
  /// </summary>
  public readonly struct NNEvaluatorResult
  {
    #region Private data

    private readonly float winP;
    private readonly float lossP;

    private readonly float win1P;
    private readonly float loss1P;

    private readonly float win2P;
    private readonly float loss2P;

    #endregion

    /// <summary>
    /// Moves left head output.
    /// </summary>
    public readonly float M;

    /// <summary>
    /// Uncertainty of value head output.
    /// </summary>
    public readonly float UncertaintyV;

    /// <summary>
    /// Uncertainty of policy head output.
    /// </summary>
    public readonly float UncertaintyP;

    /// <summary>
    /// Policy head output.
    /// </summary>
    public readonly CompressedPolicyVector Policy;

    /// <summary>
    /// Action win/draw/loss probabilities.
    /// </summary>
    public readonly CompressedActionVector ActionsWDL;

    /// <summary>
    /// Activations from certain hidden layers (optional).
    /// </summary>
    public readonly NNEvaluatorResultActivations Activations;

    /// <summary>
    ///  Optional extra evaluation statistic 0.
    /// </summary>
    public readonly FP16? ExtraStat0;

    /// <summary>
    ///  Optional extra evaluation statistic 1.
    /// </summary>
    public readonly FP16? ExtraStat1;

    /// <summary>
    /// Optional contextual information to be potentially used 
    /// as supplemental input for the evaluation of children.
    /// </summary>
    public readonly Half[] PriorState;

    /// <summary>
    /// Optional raw outputs from neural network.
    /// </summary>
    public readonly FP16[][] RawNetworkOutputs;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="winP"></param>
    /// <param name="lossP"></param>
    /// <param name="win2P"></param>
    /// <param name="loss2P"></param>
    /// <param name="m"></param>
    /// <param name="uncertaintyV"></param>
    /// <param name="uncertaintyP"></param>
    /// <param name="policy"></param>
    /// <param name="actionsWDL"></param>
    /// <param name="activations"></param>
    /// <param name="priorState"></param>
    /// <param name="extraStat0"></param>
    /// <param name="extraStat1"></param>
    public NNEvaluatorResult(float winP, float lossP, 
                             float win1P, float loss1P,
                             float win2P, float loss2P,
                             float m, float uncertaintyV, float uncertaintyP,
                             CompressedPolicyVector policy,
                             CompressedActionVector actionsWDL,
                             NNEvaluatorResultActivations activations,
                             Half[] priorState,
                             FP16? extraStat0 = null, 
                             FP16? extraStat1 = default,
                             FP16[][] rawNetworkOutputs = null)
    {
      this.winP = winP;
      this.lossP = lossP;
      this.win1P = win1P;
      this.loss1P = loss1P;
      this.win2P = win2P;
      this.loss2P = loss2P;
      ActionsWDL = actionsWDL;

      M = Math.Max(0, m);
      UncertaintyV = uncertaintyV;
      UncertaintyP = uncertaintyP;
      Policy = policy;
      Activations = activations;
      PriorState = priorState;
      ExtraStat0 = extraStat0;
      ExtraStat1 = extraStat1;
      RawNetworkOutputs = rawNetworkOutputs;
    }


    /// <summary>
    /// Value (win minus loss probability).
    /// </summary>
    public readonly float V => float.IsNaN(lossP) ? winP : (winP - lossP);


    /// <summary>
    /// Value of primary value head (win minus loss probability).
    /// </summary>
    public readonly float V1 => float.IsNaN(loss1P) ? win1P : (win1P - loss1P);


    /// <summary>
    /// Value of secondary value head (win minus loss probability).
    /// </summary>
    public readonly float V2 => float.IsNaN(loss2P) ? win2P : (win2P - loss2P);

    /// <summary>
    /// Draw probability.
    /// </summary>
    public readonly float D => 1.0f - (winP + lossP);

    /// <summary>
    /// Draw probability (secondary value head).
    /// </summary>
    public readonly float D2 => 1.0f - (win2P + loss2P);


    /// <summary>
    /// Win probability.
    /// </summary>
    public readonly float W => float.IsNaN(lossP) ? float.NaN : winP;


    /// <summary>
    /// Win probability (primary value head).
    /// </summary>
    public readonly float W1 => float.IsNaN(loss1P) ? float.NaN : winP;

    /// <summary>
    /// Win probability (secondary value head).
    /// </summary>
    public readonly float W2 => float.IsNaN(loss2P) ? float.NaN : win2P;


    /// <summary>
    /// Loss probability.
    /// </summary>
    public readonly float L => float.IsNaN(lossP) ? float.NaN : lossP;

    /// <summary>
    /// Loss probability (primary value head).
    /// </summary>
    public readonly float L1 => float.IsNaN(loss1P) ? float.NaN : lossP;

    /// <summary>
    /// Loss probability (secondary value head).
    /// </summary>
    public readonly float L2 => float.IsNaN(loss2P) ? float.NaN : loss2P;


    /// <summary>
    /// Returns most probably game result (win, draw, loss) as an integer (-1, 0, 1).
    /// </summary>
    public readonly int MostProbableGameResult => W > 0.5f ? 1 : (L > 0.5 ? -1 : 0);

    /// <summary>
    /// Returns the action head evaluation for a specified move.
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public readonly (float w, float d, float l) ActionWDLForMove(MGMove move)
      => ActionWDLForMove(ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(move));



    /// <summary>
    /// Returns the action head evaluation for a specified move.
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public readonly (float w, float d, float l) ActionWDLForMove(EncodedMove move)
    {
      int policyIndex = 0;
      foreach (var policyInfo in Policy.ProbabilitySummary())
      {
        if (policyInfo.Move == move)
        {
          (Half W, Half L) thisAction = ActionsWDL[policyIndex];
          return ((float)thisAction.W, 1 - (float)thisAction.W - (float)thisAction.L, (float)thisAction.L);
        }
        policyIndex++;
      } 

      throw new Exception("Move not found in policy " + move);
    }


    /// <summary>
    /// Returns the action win/draw/loss probabilities for the move appearing at a specified index in the compressed policy vector.
    /// </summary>
    /// <param name="moveIndexInCompressedPolicyVector"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public readonly (float w, float d, float l) ActionWDLAtCompressedPolicyIndex(int moveIndexInCompressedPolicyVector)
    {
      if (moveIndexInCompressedPolicyVector < 0 || moveIndexInCompressedPolicyVector >= Policy.Count)
      {
        throw new ArgumentOutOfRangeException(nameof(moveIndexInCompressedPolicyVector));
      }

      (Half W, Half L) thisAction = ActionsWDL[moveIndexInCompressedPolicyVector];
      return ((float) thisAction.W, 1 - (float)thisAction.W - (float)thisAction.L, (float)thisAction.L); 
    } 


    /// <summary>
    /// Returns string description.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string extraV = float.IsNaN(V2) ? "" : $" V2={V2,5:F2}";

      string dev = "";
      if (ExtraStat0 != null && !float.IsNaN(ExtraStat0.Value))
      {
        dev = $" QDEV [{ExtraStat0.Value,4:F2} {ExtraStat1.Value,4:F2}] ";
      }

      string extras = $" WDL ({W:F2} {1-(W+L):F2} {L:F2}) ";
      if (!float.IsNaN(W2))
      {
        extras += $" WDL2 ({W2:F2} {1 - (W2 + L2):F2} {L2:F2}) ";
      }

      if (!float.IsNaN(M))
      {
        extras += $" MLH={M,6:F2}";
      }

      if (!float.IsNaN(UncertaintyV))
      {
        extras += $" UV={UncertaintyV,5:F2}";
      }

      if (!float.IsNaN(UncertaintyP))
      {
        extras += $" UP={UncertaintyP,5:F2}";
      }

      return $"<NNPositionEvaluation V={V,6:F2}{extraV}{extras}{dev} Policy={Policy}>";
    }
  }
}
