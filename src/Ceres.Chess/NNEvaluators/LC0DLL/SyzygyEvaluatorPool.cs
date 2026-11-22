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
using Ceres.Chess.TBBackends.Fathom;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.Chess.NNEvaluators.LC0DLL
{
  /// <summary>
  /// Maintains a set of evaluators (one for each distinct path to tablebases).
  /// </summary>
  public static class SyzygyEvaluatorPool
  {
    const int MAX_SESSIONS = 32; // hardcoded in C++
    static IDPool sessionIDPool = new IDPool("SyzygyEvaluator", MAX_SESSIONS);

    /// <summary>
    /// Evaluator (and its session ID) currently loaded for each set of paths.
    /// </summary>
    static Dictionary<string, (ISyzygyEvaluatorEngine Evaluator, int SessionID)> pathsToEvaluatorDict = new ();

    public static Func<ISyzygyEvaluatorEngine> OverrideEvaluatorFactory;

    /// <summary>
    /// Returns a session for the specified paths using default settings (from Ceres.json).
    /// </summary>
    /// <param name="paths"></param>
    /// <returns></returns>
    public static ISyzygyEvaluatorEngine GetSessionForPaths(string paths)
    {
      ArgumentNullException.ThrowIfNull(paths);

      lock (sessionIDPool)
      {
        if (pathsToEvaluatorDict.TryGetValue(paths, out (ISyzygyEvaluatorEngine Evaluator, int SessionID) existing))
        {
          return existing.Evaluator;
        }

        // The backend holds process-wide state for a single set of paths (FathomTB keeps one
        // static probe and throws on reinitialization), so a request for different paths must
        // first release whatever is loaded rather than let that initialization throw.
        //
        // N.B. this invalidates any evaluator previously handed out for the old paths. Callers
        // re-request per search (see EvaluatorSyzygy), so switching between searches is safe;
        // switching while a search still using the old paths is running is not.
        ReleaseAllInternal();

        int sessionID = sessionIDPool.GetFreeID();
        ISyzygyEvaluatorEngine evaluator = OverrideEvaluatorFactory != null ? OverrideEvaluatorFactory()
                                                                           : new FathomEvaluator();
        try
        {
          evaluator.Initialize(paths);
        }
        catch
        {
          // Leave no session ID stranded if the paths turn out to be unusable.
          sessionIDPool.ReleaseID(sessionID);
          throw;
        }

        pathsToEvaluatorDict[paths] = (evaluator, sessionID);
        return evaluator;
      }
    }


    /// <summary>
    /// Releases all pooled evaluators, unloading their tablebases and freeing the
    /// underlying file mappings. Any evaluator reference previously returned by
    /// GetSessionForPaths becomes invalid; callers must re-request one afterward.
    /// </summary>
    public static void ReleaseAll()
    {
      lock (sessionIDPool)
      {
        ReleaseAllInternal();
      }
    }


    /// <summary>
    /// Disposes and forgets every pooled evaluator. Caller must hold the pool lock.
    /// </summary>
    static void ReleaseAllInternal()
    {
      foreach ((ISyzygyEvaluatorEngine evaluator, int sessionID) in pathsToEvaluatorDict.Values)
      {
        evaluator.Dispose();
        sessionIDPool.ReleaseID(sessionID);
      }

      pathsToEvaluatorDict.Clear();
    }

  }
}
