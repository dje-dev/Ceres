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

namespace Ceres.Chess.NNEvaluators.Ceres.TPG
{
  /// <summary>
  /// Set of target training values to be emitted into TPG file 
  /// (everything except policy).
  /// </summary>
  public record struct TPGTrainingTargetNonPolicyInfo
  {
    public enum TargetSourceInfo
    {
      /// <summary>
      /// Ordinary position encountered during game play.
      /// </summary>
      Training,

      /// <summary>
      /// Position was explicitly marked as blunder (non-optimal move deliberately chosen in game play).
      /// </summary>
      NoiseDeblunder,

      /// <summary>
      /// Position was determined to be a blunder based on subsequent game play.
      /// </summary>
      UnintendedDeblunder,

      /// <summary>
      /// Rescored endgame position from tablebase.
      /// </summary>
      Tablebase,

      /// <summary>
      /// If the position was not in the training data but instead
      /// inserted as a random move that will be used only as a source of action head information
      /// (but not with any value/policy target available).
      /// </summary>
      ActionHeadDummyMove,
    };


    /// <summary>
    /// The game result after deblundering is applied.
    /// </summary>
    public (float w, float d, float l) ResultDeblunderedWDL;

    /// <summary>
    /// The game result without deblundering.
    /// </summary>
    public (float w, float d, float l) ResultNonDeblunderedWDL;

    /// <summary>
    /// Search value score (Q) from training game.
    /// </summary>
    public (float w, float d, float l) BestWDL;

    /// <summary>
    /// Moves left (in plies).
    /// </summary>
    public float MLH;

    /// <summary>
    /// Uncertainty of value score (absolute difference between value head V and search Q).
    /// </summary>
    public float DeltaQVersusV;

    /// <summary>
    /// Uncertainty of policy score (KLD between policy head and search policy).
    /// </summary>
    public float KLDPolicy;

    /// <summary>
    /// Largest downside Q deviation for remaining moves in game.
    /// </summary>
    public float ForwardMinQDeviation;

    /// <summary>
    /// Largest upside Q deviation for remaining moves in game.
    /// </summary>
    public float ForwardMaxQDeviation;

    /// <summary>
    /// Search evaluation forward an intermediate number of moves in game.
    /// </summary>
    public (float, float, float) IntermediateWDL;

    /// <summary>
    /// Short-term expected absolute change in Q from this position.
    /// </summary>
    public float DeltaQForwardAbs;

    /// <summary>
    /// Source of training data record.
    /// </summary>
    public TargetSourceInfo Source;

    /// <summary>
    /// Estimated Q by which the move played to get this position was 
    /// worse than the best move available at the parent.
    /// </summary>
    public float PlayedMoveQSuboptimality;

    /// <summary>
    /// Number of search nodes in the search tree.
    /// </summary>
    public int NumSearchNodes;

    /// <summary>
    /// Sum of Q of all injected favorable training game blunders from here to end of game.
    /// </summary>
    public float ForwardSumPositiveBlunders;

    /// <summary>
    /// Sum of Q of all injected unfavorable training game blunders from here to end of game.
    /// </summary>
    public float ForwardSumNegativeBlunders;

    /// <summary>
    /// Value head estimate of win probability (from our perspective) at the prior position.
    /// </summary>
    public float PriorPositionWinP;

    /// <summary>
    /// Value head estimate of draw probability (from our perspective) at the prior position.
    /// </summary>
    public float PriorPositionDrawP;

    /// <summary>
    /// Value head estimate of loss probability (from our perspective) at the prior position.
    /// </summary>
    public float PriorPositionLossP;

    /// <summary>
    /// Neural net index (0...1857) of the move played from prior move in game (or -1 if none).
    /// </summary>
    public short PolicyIndexInParent;
  }

}
