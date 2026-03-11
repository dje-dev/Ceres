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
    /// Array of names of raw neural network outputs (if RetainRawOutputs is true).
    /// </summary>
    public readonly string[] RawNetworkOutputNames;

    /// <summary>
    /// Per-square ply-bin move probabilities (512 elements: 64 squares * 8 bins), or null.
    /// </summary>
    public readonly Half[] PlyBinMoveProbs;

    /// <summary>
    /// Per-square ply-bin capture probabilities (512 elements: 64 squares * 8 bins), or null.
    /// </summary>
    public readonly Half[] PlyBinCaptureProbs;

    /// <summary>
    /// PUNIM self probabilities (8 elements: 8 bins), or null.
    /// </summary>
    public readonly Half[] PunimSelfProbs;

    /// <summary>
    /// PUNIM opponent probabilities (8 elements: 8 bins), or null.
    /// </summary>
    public readonly Half[] PunimOpponentProbs;

    /// <summary>
    /// Square index (0-63) of the side-to-move's king.
    /// Used for VCapture calculation from PlyBinCaptureProbs.
    /// </summary>
    private readonly byte ourKingSquare;

    /// <summary>
    /// Square index (0-63) of the opponent's king.
    /// Used for VCapture calculation from PlyBinCaptureProbs.
    /// </summary>
    private readonly byte theirKingSquare;

    /// <summary>
    /// Fortress probability metric: minimum (1 - P(NEVER)) over all pawn squares.
    /// Low values indicate a pawn unlikely to ever move, suggesting fortress-like structure.
    /// Returns NaN if PlyBinMoveProbs unavailable or no pawns on board.
    /// </summary>
    public readonly float FortressP;


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
    /// <param name="ourKingSquare">Square index of the side-to-move's king (for VCapture).</param>
    /// <param name="theirKingSquare">Square index of the opponent's king (for VCapture).</param>
    /// <param name="fortressP">Fortress probability metric (min pChange over pawn squares).</param>
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
                             FP16[][] rawNetworkOutputs = null,
                             string[] rawNetworkOutputNames = null,
                             Half[] plyBinMoveProbs = null,
                             Half[] plyBinCaptureProbs = null,
                             Half[] punimSelfProbs = null,
                             Half[] punimOpponentProbs = null,
                             byte ourKingSquare = 0,
                             byte theirKingSquare = 0,
                             float fortressP = float.NaN)
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
      RawNetworkOutputNames = rawNetworkOutputNames;
      PlyBinMoveProbs = plyBinMoveProbs;
      PlyBinCaptureProbs = plyBinCaptureProbs;
      PunimSelfProbs = punimSelfProbs;
      PunimOpponentProbs = punimOpponentProbs;
      this.ourKingSquare = ourKingSquare;
      this.theirKingSquare = theirKingSquare;
      FortressP = fortressP;
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
    /// Value derived from ply-bin capture probabilities on king squares.
    /// 
    /// VCapture = P(we capture their king) - P(they capture our king)
    /// 
    /// Positive values indicate we are more likely to deliver checkmate than receive one.
    /// Returns NaN if capture probabilities are not available.
    /// </summary>
    public readonly float VCapture
    {
      get
      {
        if (PlyBinCaptureProbs == null)
        {
          return float.NaN;
        }

        const int NUM_BINS = 8;
        const int NEVER_CAPTURED_BIN = 7;

        // Read "never captured" probability (bin 7) for each king.
        // Array layout: 64 squares × 8 bins = 512 elements, indexed as [square * 8 + bin].
        float pNeverOurKing = (float)PlyBinCaptureProbs[ourKingSquare * NUM_BINS + NEVER_CAPTURED_BIN];
        float pNeverTheirKing = (float)PlyBinCaptureProbs[theirKingSquare * NUM_BINS + NEVER_CAPTURED_BIN];

        // Convert to capture probabilities.
        float selfCaptureProb = 1f - pNeverTheirKing;     // P(we eventually capture their king)
        float opponentCaptureProb = 1f - pNeverOurKing;   // P(they eventually capture our king)

        return selfCaptureProb - opponentCaptureProb;
      }
    }


    /// <summary>
    /// Computes the FortressP metric from ply-bin move probabilities and position.
    /// FortressP = min(P(NEVER)) over all squares containing pawns.
    /// A high value indicates a pawn unlikely to ever move, suggesting a fortress-like structure.
    /// </summary>
    /// <param name="plyBinMoveProbs">512-element array (64 squares × 8 bins) of move probabilities.</param>
    /// <param name="pos">The position (MGPosition) to analyze for pawn squares.</param>
    /// <param name="blackToMove">True if black is to move (requires rank-flip for NN output perspective).</param>
    /// <returns>Minimum P(NEVER) over all pawn squares, or 0 if no pawns, or NaN if probs unavailable.</returns>
    public static float ComputeFortressP(ReadOnlySpan<Half> plyBinMoveProbs, in MGPosition pos, bool blackToMove)
    {
      const bool DEBUG_OUTPUT_FORTRESS = false;

      if (plyBinMoveProbs.IsEmpty)
      {
        return float.NaN;
      }

      const int NUM_BINS = 8;
      const int NEVER_MOVE_BIN = 7;

      // Extract pawn bitboards using MGPosition encoding:
      // Pawns: A=1, B=0, C=0 (piece type 001). D distinguishes color (D=0 white, D=1 black).
      ulong whitePawns = ~pos.D & ~pos.C & ~pos.B & pos.A;
      ulong blackPawns = pos.D & ~pos.C & ~pos.B & pos.A;
      ulong allPawns = whitePawns | blackPawns;

      if (allPawns == 0)
      {
        // No pawns on board - return 0 (not NaN) per specification.
        return 0f;
      }

      if (DEBUG_OUTPUT_FORTRESS)
      {
        Console.WriteLine($"FortressP computation (blackToMove={blackToMove}):");
      }

      float minPNever = float.MaxValue;

      while (allPawns != 0)
      {
        // Get next pawn square in MGPosition bit-order (H1=0, file-mirrored).
        int mgSquare = System.Numerics.BitOperations.TrailingZeroCount(allPawns);
        allPawns &= allPawns - 1; // Clear lowest bit

        // Convert MGPosition square to standard square (A1=0).
        // MGPosition: H1=0, bit index = rank*8 + (7-file), so file = 7 - (mgSquare % 8).
        int rank = mgSquare / 8;
        int file = 7 - (mgSquare % 8);
        int stdSquare = rank * 8 + file;

        // Rank-flip for black-to-move (NN outputs in side-to-move perspective).
        if (blackToMove)
        {
          stdSquare ^= 56;
        }

        // Read P(NEVER move) for this square.
        float pNever = (float)plyBinMoveProbs[stdSquare * NUM_BINS + NEVER_MOVE_BIN];

        if (DEBUG_OUTPUT_FORTRESS)
        {
          bool isWhitePawn = (whitePawns & (1UL << mgSquare)) != 0;
          string pawnColor = isWhitePawn ? "White" : "Black";
          char fileChar = (char)('a' + file);
          int rankNum = rank + 1;
          Console.WriteLine($"  {pawnColor} pawn at {fileChar}{rankNum}: P(NEVER) = {pNever * 100:F1}%");
        }

        if (pNever < minPNever)
        {
          minPNever = pNever;
        }
      }

      if (DEBUG_OUTPUT_FORTRESS)
      {
        Console.WriteLine($"  --> FortressP (min P(NEVER)) = {minPNever * 100:F1}%");
      }

      return minPNever;
    }


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
    public readonly float W => float.IsNaN(winP) ? float.NaN : winP;


    /// <summary>
    /// Win probability (primary value head).
    /// </summary>
    public readonly float W1 => float.IsNaN(win1P) ? float.NaN : win1P;

    /// <summary>
    /// Win probability (secondary value head).
    /// </summary>
    public readonly float W2 => float.IsNaN(win2P) ? float.NaN : win2P;


    /// <summary>
    /// Loss probability.
    /// </summary>
    public readonly float L => float.IsNaN(lossP) ? float.NaN : lossP;

    /// <summary>
    /// Loss probability (primary value head).
    /// </summary>
    public readonly float L1 => float.IsNaN(loss1P) ? float.NaN : loss1P;

    /// <summary>
    /// Loss probability (secondary value head).
    /// </summary>
    public readonly float L2 => float.IsNaN(loss2P) ? float.NaN : loss2P;


    /// <summary>
    /// Returns most probable game result (win, draw, loss) as an integer (1, 0, -1).
    /// </summary>
    public readonly int MostProbableGameResult => W > 0.5f ? 1 : (L > 0.5f ? -1 : 0);

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

      throw new ArgumentException($"Move {move} not found in policy.", nameof(move));
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
      return ((float)thisAction.W, 1 - (float)thisAction.W - (float)thisAction.L, (float)thisAction.L);
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

      string extras = $" WDL ({W:F2} {D:F2} {L:F2}) ";
      if (!float.IsNaN(W2))
      {
        extras += $" WDL2 ({W2:F2} {D2:F2} {L2:F2}) ";
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

      if (!float.IsNaN(VCapture))
      {
        extras += $" VC={VCapture,5:F2}";
      }

      if (!float.IsNaN(FortressP))
      {
        extras += $" FP={FortressP,5:F2}";
      }

      return $"<NNPositionEvaluation V={V,6:F2}{extraV}{extras}{dev} Policy={Policy}>";
    }
  }
}
