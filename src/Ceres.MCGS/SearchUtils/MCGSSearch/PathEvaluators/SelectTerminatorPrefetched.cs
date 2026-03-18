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
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Paths;
using System.Threading;


#endregion

namespace Ceres.MCGS.Search.PathEvaluators;

/// <summary>
/// Checks for nodes that have already been prefetched in the graph,
/// i.e. their value has been prefetched and stored in the graph
/// (but not yet visited).
/// </summary>
public sealed class SelectTerminatorPrefetched : SelectTerminatorBase
{
  internal bool InPrefetchMode;

  /// <summary>
  /// Total number of nodes selected from prefetched nodes.
  /// </summary>
  public static long NumNodesSelectedFromPrefetch;


  /// <summary>
  /// Constructor.
  /// </summary>
  public SelectTerminatorPrefetched()
  {
  }


  protected override bool DoTryTerminate(MCGSPath path, ref SelectTerminationInfo terminationInfo)
  {
    GNode lastNode = path.LeafVisitRef.ParentChildEdge.ChildNode;

    ref readonly GNodeStruct nodeRef = ref lastNode.NodeRef;
    
    // Do not terminate on this node if already evaluated (allow further descent).
    if (InPrefetchMode && nodeRef.IsEvaluated)
    {
      return false;
    }
      
    // 
    if (lastNode.NodeRef.N == 0
     && !FP16.IsNaN(lastNode.NodeRef.WinP))
    {
      // This node has been prefetched with a value.
      // We can use it.
      FP16 mlh = 0; // GFIX: TODO: this seems not to have been stored in the node? Recover it and use below.
      terminationInfo = new(lastNode.NodeRef.IsWhite ? SideType.White : SideType.Black, 
                            MCGSPathTerminationReason.PendingNeuralNetEval,
                            nodeRef.Terminal, nodeRef.WinP, nodeRef.LossP, mlh, 
                            nodeRef.UncertaintyValue, nodeRef.UncertaintyPolicy);

      if (!terminationInfo.GameResult.IsTerminal())
      {
        Interlocked.Increment(ref NumNodesSelectedFromPrefetch);
      }

      return true;
    }
    else
    {
      return false;
    }
  }


}
