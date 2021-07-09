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

using System;

namespace Ceres.Features
{
  /// <summary>
  /// Parses command line arguments relating to 
  /// child worker processes (if present).
  /// 
  /// Argument specification is of the form:
  ///   WORKER.NUMA_ID.GPU_ID 
  /// such as "WORKER.0.0"
  /// </summary>
  public static class CommandLineWorkerSpecification
  {
    public static bool IsWorker { get; private set; }
    public static int? NumaNodeID { get; private set; }
    public static int GPUID { get; private set; }

    static CommandLineWorkerSpecification()
    {
      // Start with defaults.
      NumaNodeID = 0;
      GPUID = 0;

      IsWorker = false;
      foreach (string argument in Environment.GetCommandLineArgs())
      {
        if (argument.ToUpper().StartsWith("WORKER"))
        {
          IsWorker = true;

          // Parse optional numa node and GPU id
          string[] workerParts = argument.Split(".");
          if (workerParts.Length > 1)
          {
            // Asterist indicates no NUMA binding.
            NumaNodeID = workerParts[1] == "*" ? null : int.Parse(workerParts[1]);
          }
          if (workerParts.Length > 2)
          {
            GPUID = int.Parse(workerParts[2]);
          }
        }
      }
    }

  }
}