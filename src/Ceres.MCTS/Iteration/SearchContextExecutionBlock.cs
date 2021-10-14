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

using Ceres.Base.Environment;
using Ceres.Chess;
using Ceres.MCTS.MTCSNodes.Storage;
using System;
using System.Threading;

#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Structure that sets the ambient MCTSManager.ThreadSearchContext 
  /// for this thread upon construction and releases it upon IDispose.
  /// 
  /// It is intended that operations upon an MCTSNodeStore will
  /// enclose code within a using block which creates such an object,
  /// thereby setting/unsetting the context as appropriate
  /// (also with nesting supported).
  /// </summary>
  public struct SearchContextExecutionBlock : IDisposable
  {
    public readonly MCTSIterator Context;
    MCTSIterator priorContext;
    MCTSNodeStoreContext storeContext;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="context"></param>
    public SearchContextExecutionBlock(MCTSIterator context)
    {
      priorContext = MCTSManager.ThreadSearchContext;
      Context = context;
      storeContext = new MCTSNodeStoreContext(context.Tree.Store);
      MCTSManager.ThreadSearchContext = context;

      // Pass the value of the test flag for this context
      // down to the ThreadStatic variable in the Base project
      // so code in lower levels (Base/Chess) can possibly observe it.
      CeresEnvironment.TEST_MODE =  context.ParamsSearch.TestFlag;
    }


    /// <summary>
    /// Dispose method which deactivates the ambient context.
    /// </summary>
    public void Dispose()
    {
      storeContext.Dispose();
      storeContext = default;
      MCTSManager.ThreadSearchContext = priorContext;
    }
  }
}
