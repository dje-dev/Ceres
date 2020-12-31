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

using Ceres.Base.Math;
using System;

#endregion

namespace Ceres.MCTS.Search
{
  /// <summary>
  /// Manager for batch collection strategy during search,
  /// dynamically calculating any adjustments to
  /// batch size or virtual loss magnitude that may be optimal.
  /// 
  /// NOTE: Tests showed this successful at increasing node yield 
  ///       in positions with high numbers of collisions.
  ///       However it was unclear that play quality was improved,
  ///       and it seemingly came at about a 5% reduction in search speed.
  ///       This is enabled by setting ParamsSelect.UseDynamicVLoss to true.
  /// </summary>
  internal class MCTSBatchParamsManager
  {
    public readonly bool DynamicBatchingEnabled;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="dynamicBatchingEnabled"></param>
    public MCTSBatchParamsManager(bool dynamicBatchingEnabled)
    {
      DynamicBatchingEnabled = dynamicBatchingEnabled;
    }


    public float VLossDynamicBoostForSelector()
    {
      if (!DynamicBatchingEnabled) return 0;

      if (lastSelectorYield < 0.6) return -0.10f;
    
      return 0f;
    }

    float lastBoost = 1.0f;

    public float BatchSizeDynamicScaleForSelector()
    {
      if (!DynamicBatchingEnabled) return 1;

      if (lastSelectorYield < 0.6) lastBoost *= 0.8f;
      if (lastSelectorYield > 0.8) lastBoost *= 1.2f;

      lastBoost = MathHelpers.Bounded(lastBoost, 0.4f, 1.0f);
     
      return lastBoost;
    }

    float lastSelectorYield = 1.0f;


    // --------------------------------------------------------------------------------------------
    public void UpdateVLossDynamicBoost(int numLeafsAttempted, float yieldThisBatch)
    {
      if (numLeafsAttempted > 20)
      {
          lastSelectorYield = yieldThisBatch;
      }

      //      Console.WriteLine("("+selectorID + ") " + numLeafsAttempted + " yields " + yieldThisBatch);
#if NOT
    float lastCollisionFraction = 0;

    bool ENABLE_ADJUST_VLOSS => context.ParamsSearch.VLOSS_ADJUST;

      float collisionFrac = 0;
      if (numNewCollisions > 0) // TODO: why needed?
      {
        collisionFrac = numNewCollisions / numTargetLeafs;
        lastCollisionFraction = collisionFrac;
      }

      const float THRESHOLD_ADJUST = 0.10f;
      if (numTargetLeafs > 16 && collisionFrac > THRESHOLD_ADJUST)
      {
        const float SCALING_FACTOR = 5.0f;
        if (ENABLE_ADJUST_VLOSS)
        {
          vLossDynamicBoost = -SCALING_FACTOR * (collisionFrac - THRESHOLD_ADJUST);
        }
//        Console.WriteLine(collisionFrac + " " + numTargetLeafs + " " + numNewCollisions);
      }
      else
        vLossDynamicBoost = 0.0f;
#endif

    }

  }
}


#if nOT
          if (nodesNN.Count > 10)
          {
            MCTSNode node = nodesNN[^1];
            if (selectorID == 0)
              node.Ref.BackupAbortUpdate(node.NInFlight, 0);
            else
              node.Ref.BackupAbortUpdate(0, node.NInFlight2);
            nodesNN.RemoveAt(nodesNN.Count - 1);
          }
#endif
#if NOT
          // TODO: cleanup or delete
          SortedList<double, MCTSNode> sl = new SortedList<double, MCTSNode>(selectedNodes.Count);
          double rand = 0;
          //Random r = new Random();
          foreach (MCTSNode node in selectedNodes)
          {
            float priorityValue = selectorID == 0 ? node.NodeSelector0PriorityFactor
                                                  : node.NodeSelector1PriorityFactor;
            float adjustedPriorityValue = MathF.Pow(priorityValue, 1.0f / node.Depth);
            sl.Add(adjustedPriorityValue + rand, node);
            rand += 0.00001;//= r.NextDouble() * 0.00001;
          }
          ListBounded<MCTSNode> newNodes = new ListBounded<MCTSNode>(selectedNodes.Count);
          for (int i = 0; i < sl.Count; i++)
          {
            MCTSNode node = sl.Values[i];
            if (sl.Count < 5 || i < sl.Count / 10)
              newNodes.Add(node);
            else
            {
              node.Ref.BackupAbortUpdate(node.NInFlight, node.NInFlight2);
              //node.ActionType = MCTSNode.NodeActionType.None;
            }

          }
          selectedNodes = newNodes;
#endif