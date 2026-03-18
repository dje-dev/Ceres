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
using System.Diagnostics;
using System.Runtime.InteropServices;

using Ceres.Base.Misc;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Phases;

#endregion

namespace Ceres.MCGS.Search.Paths;

/// <summary>
/// Represents a visit to a node during Monte Carlo Graph Search visit to a leaf node, comprising:
///   - a parent node which was the node at which a child was selected to continue descending
///   - a child node that was selected
///   - an edge connecting the parent to the child
///  
/// These path visits are used to capture information about MCGS visits sufficient to
/// perform backup operations after a leaf is selected and evaluated. They also allow
/// search to proceed recursively without passing along many state variables in the argument list
/// since they can be obtained by consulting the prior path visit.
/// 
/// The first visit in a path has parent of the root.
///
/// There is only one special case. The last visit in a path may select an edge which is terminal,
/// meaning that there is no corresponding materialized child node in the graph.  
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 2)]
public struct MCGSPathVisit
{
  #region Containing path information

  /// <summary>
  /// Containing path.
  /// </summary>
  public MCGSPath ParentPath;

  /// <summary>
  /// The edge connecting the parent node to the child node.
  /// </summary>
  public GEdge ParentChildEdge;

  #endregion

  /// <summary>
  /// If backups should "disconnect" starting at this visit,
  /// meaning that beginning with the edge & parent node of this visit
  /// there are no further to any stats (e.g. Q or N) for any node
  /// higher in the graph (instead only the InFlight statistics backed out).
  /// 
  /// NOTE: not currently set in any code path, consider converting to another or unused field.
  /// </summary>
  public bool DisconnectFromEdgeNStartingThisVisit;


  #region Path visit definition

  /// <summary>
  /// Index of the chosen MCGS child for this path (from the parent node).
  /// </summary>
  public short IndexOfChildInParent;

  #endregion

  #region Associated data

  /// <summary>
  /// Number of visits attempted to this node (set during selection phase).
  /// </summary>
  public int NumVisitsAttempted;

  /// <summary>
  /// Number of visits which were actually accepted (determined during backup phase).
  /// Recording this here serves at least two purposes:
  ///   - documents actual updates applied to graph for one iteration (for transparency/debugging purposes)
  ///   - when suspending a path to be resumed later (during reduction)
  ///     serves as a place to remember how many visits were actually allocated to be accepted upon resumption. 
  /// </summary>
  public short? NumVisitsAccepted;


  /// <summary>
  /// Hash of child node (not including any accumulated history).
  /// </summary>
  public PosHash64 ChildNodeHashStandalone64;

  /// <summary>
  /// If the move that transitioned from parent to child node
  /// was irreversible.
  /// </summary>
  public bool MoveIrreverisible;


  #endregion

  /// <summary>
  /// Position corresponding to the parent node.
  /// </summary>
  public MGPosition ChildPosition;

  /// <summary>
  /// List of legal moves from the position.
  /// (possibly null; created and cached on-the-fly by Moves if necessary).
  /// </summary>
  internal MGMoveList MovesList;

  /// <summary>
  /// Number of policy moves from neural network.
  /// Used as a sizing hint about probable number of legal moves.
  /// </summary>
  private short NumPolicyMoves;


  #region Reduction Mode Fields

  /// <summary>
  /// If node is multi-child, holds the aggregated backup values
  /// accumulated across the multiple children.
  /// </summary>
  public MCGSBackupAccumulator Accumulator;

  /// <summary>
  /// Counter number of paths which pass thru this node and
  /// are have yet to be backed up (inclusive of all parents).
  /// Note that for efficiency reasons, the backup phases decrements
  /// this counter only if the node turned out to be a merge node (NumPathsVisitedBy > 1).
  /// </summary>
  public volatile int NumVisitsAttemptedPendingBackup;

  #endregion


  /// <summary>
  /// List of legal moves from the position.
  /// </summary>
  public MGMoveList Moves => MovesList ??= MGMoveGen.GeneratedMoves(in ChildPosition);


  /// <summary>
  /// Forces generation of moves if not already done.
  /// </summary>
  public void ForceMoveGeneration()
  {
    // Accessing Moves property will generate and cache moves if not already done.
    _ = Moves;
  }


  /// <summary>
  /// If Moves have already been generated and cached.
  /// </summary>
  public readonly bool MovesAreGenerated => MovesList != null;


  /// <summary>
  /// 
  /// TODO: Used only for temporary special case of root node.
  /// Someday remove this, by decoupling NN evaluation from paths (SelectTerminatorNeuralNetwork rework).
  /// </summary>
  internal readonly bool IsRootInitializationPath => IndexOfChildInParent == -1;


  /// <summary>
  /// Returns the parent node of this path visit.
  /// </summary>
  public readonly GNode ChildNode => IsRootInitializationPath ? ParentPath.Engine.SearchRootNode : ParentChildEdge.ChildNode;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="parentPath"></param>
  /// <param name="parentNode"></param>
  /// <param name="childEdgeIndex"></param>
  /// <param name="childStandaloneHash"></param>
  public MCGSPathVisit(MCGSPath parentPath, GNode parentNode, int childEdgeIndex, 
                       PosHash64 childStandaloneHash64)
  {
    ParentPath = parentPath;
    ParentChildEdge = childEdgeIndex == -1 ? default : parentNode.ChildEdgeAtIndex(childEdgeIndex);
    IndexOfChildInParent = (short)childEdgeIndex;
    ChildNodeHashStandalone64 = childStandaloneHash64;
  }


  static uint typeHashValue;
  static bool TypeFieldsHaveChanged()
  {
    const uint EXPECTED_HASH = 367620873;

    if (typeHashValue == default)
    {
      typeHashValue = ObjUtils.CalcTypeLayoutHash(typeof(MCGSPathVisit));
      bool matches = typeHashValue == EXPECTED_HASH;
      if (!matches)
      {
        throw new Exception($"MCGSPathVisit type layout has changed! Expected hash: { EXPECTED_HASH } vs actual { typeHashValue}");  
      }
    }

    return true;
  }


  /// <summary>
  /// Initializes the path visit with comprehensive information.
  /// </summary>
  public void Init(MCGSPath parentPath,
                   GNode parentOfThisVisitNode,
                   int childIndexOfThisVisitInParent,
                   PosHash64 currentVisitNodeHashStandalone64,
                   int numVisitsAttemptedThisVisit,
                   bool moveIrreversible,
                   in MGPosition currentVisitNodePosition)
  {
    // N.B. Every field must be explicitly reassigned because the 
    //      slots are reused and could start with visits from other batches.
    // (disabled due to unstable hash)
    // Debug.Assert(TypeFieldsHaveChanged());

    ParentPath = parentPath;
    IndexOfChildInParent = (short)childIndexOfThisVisitInParent;
    ChildNodeHashStandalone64 = currentVisitNodeHashStandalone64;
    MoveIrreverisible = moveIrreversible;
    NumVisitsAccepted = 0;
    NumVisitsAttempted = (short)numVisitsAttemptedThisVisit;
    NumVisitsAttemptedPendingBackup = NumVisitsAttempted;

    ChildPosition = currentVisitNodePosition;

    DisconnectFromEdgeNStartingThisVisit = false;
    MovesList = null;
    Accumulator = default;
    ParentChildEdge = default;

    ChildPosition = currentVisitNodePosition;
    NumPolicyMoves = parentOfThisVisitNode.NumPolicyMoves;

    // Possibly update MaxQSuboptimality
    if (MCGSParamsFixed.UPDATE_MAXQ_SUBOPTIMALITY)
    {
      if (!parentOfThisVisitNode.IsNull && IndexOfChildInParent < parentOfThisVisitNode.NumEdgesExpanded)
      {
        double parentQ = parentOfThisVisitNode.Q;
        GEdge edge = parentOfThisVisitNode.ChildEdgeAtIndex(IndexOfChildInParent);
        if (edge.N > 0)
        {
          double ourQ = edge.Q;
          float suboptimality = (float)-(parentQ + ourQ);
          parentPath.MaxQSubOptimality = MathF.Max(suboptimality, parentPath.MaxQSubOptimality);
        }
      }
    }
  }


  /// <summary>
  /// Returns string representation.
  /// </summary>
  /// <param name="shortForm"></param>
  /// <returns></returns>
  public readonly string ToString(bool shortForm)
  {
    if (ChildNodeHashStandalone64 == default)
    {
      return "(none)";
    }

    int slotInParent = -1;
    if (ParentPath != null && ParentChildEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
    {
      for (int i = 0; i < ParentPath.numSlotsUsed; i++)
      {
        if (ParentPath.slots[i].ChildNode == this.ChildNode)
        {
          slotInParent = i;
          break;
        }
      }
    }
    string slotStr = $" par: {(ParentPath != null ? ParentPath.PathID.ToString() : "N/A")} sl: {slotInParent}";

    string refInfoStr = "";

    if (!shortForm)
    {
      refInfoStr = $"pBup={NumVisitsAttemptedPendingBackup} accAtt={Accumulator.NumVisitsAttempted} accAcc={Accumulator.NumVisitsAccepted}";
    }

    string positionFen = ChildPosition == default ? "NoPos" : ChildPosition.ToPosition.FEN;
    string movesStr = MovesList == null ? "null" : MovesList.NumMovesUsed.ToString();

    return $"<MCGSPathVisit #{ParentChildEdge.ToString(shortForm)} {slotStr} Q={ParentChildEdge.Q,5:F2} "
         + $"Sib={100*ParentChildEdge.ParentNode.NodeRef.SiblingsQFrac}% "
//         + $"QD={QDistanceFromBest,5:F3} QDC={QDistanceFromBestCumulative,5:F2} QDW={QDistanceFromBestWorst,5:F3} {(IsBullet?"*B" : "")} "
         + $"att/acc: {NumVisitsAttempted}/{NumVisitsAccepted} H={ChildNodeHashStandalone64.Hash % 10_000} {refInfoStr} {positionFen} Mv={movesStr}>";
  }


  /// <summary>
  /// Returns string representation of the path visit.
  /// </summary>
  /// <returns></returns>
  public readonly override string ToString() => ToString(false);

  public readonly override bool Equals(object obj) => obj != null
                                                   && ((MCGSPathVisit)obj).ParentPath == ParentPath
                                                   && ((MCGSPathVisit)obj).ParentChildEdge == ParentChildEdge;

  public readonly override int GetHashCode() => HashCode.Combine(ParentPath.PathID.GetHashCode(), 
                                                                 ParentChildEdge.ParentNode.GetHashCode());
  
  public static bool operator ==(in MCGSPathVisit left, in MCGSPathVisit right) => left.Equals(right);

  public static bool operator !=(in MCGSPathVisit left, in MCGSPathVisit right) => !left.Equals(right);
}
