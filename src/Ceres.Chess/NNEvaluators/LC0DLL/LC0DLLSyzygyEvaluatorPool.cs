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

#endregion

namespace Ceres.Chess.NNEvaluators.LC0DLL
{
  /// <summary>
  /// Maintains a set of evaluators (one for each distinct path to tablebases).
  /// </summary>
  public static class LC0DLLSyzygyEvaluatorPool
  {
    const int MAX_SESSIONS = 32; // hardcoded in C++
    static IDPool sessionIDPool = new IDPool("LC0DLLSyzygyEvaluator", MAX_SESSIONS);

    static Dictionary<string, ISyzygyEvaluatorEngine> pathsToEvaluatorDict = new ();

    public static Func<ISyzygyEvaluatorEngine> OverrideEvaluatorFactory;

    public static ISyzygyEvaluatorEngine GetSessionForPaths(string paths)
    {
      lock (sessionIDPool)
      {
        ISyzygyEvaluatorEngine evaluator;
        if (pathsToEvaluatorDict.TryGetValue(paths, out evaluator))
        {
          return evaluator;
        }
        else
        {
          int sessionID = sessionIDPool.GetFreeID();
          evaluator = OverrideEvaluatorFactory == null ? new LC0DLLSyzygyEvaluator(sessionID) 
                                                       : OverrideEvaluatorFactory();
          evaluator.Initialize(paths);
          pathsToEvaluatorDict[paths] = evaluator;
          return evaluator;
        }
      }
    }

  }
}
