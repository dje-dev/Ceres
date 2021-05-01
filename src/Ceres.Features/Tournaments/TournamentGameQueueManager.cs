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
using System.IO;
using System.Threading;
using Ceres.Base.DataType;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Manages a queue (consisting of a set of files in a directory)
  /// used to coordinate activity of tournament coordinator 
  /// and child worker processes.
  /// </summary>
  public class TournamentGameQueueManager
  {
    /// <summary>
    /// Indicates if this manager is running as the root coordinator process
    /// that enqueues game requests and waits for worker processes to run and return results.
    /// 
    /// </summary>
    public bool IsCoordinator { get; private set; }


    /// <summary>
    /// Directory in which coordination files are placed
    /// (request and results).
    /// </summary>
    public readonly string QueueDirectory;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="queueDirectory"></param>
    public TournamentGameQueueManager(string queueDirectory)
    {
      if (queueDirectory != null)
      {
        QueueDirectory = queueDirectory;
        IsCoordinator = false;
      }
      else
      {
        QueueDirectory = Path.Combine(Path.GetTempPath(), "_TournQueue_" + DateTime.Now.Ticks);
        Directory.CreateDirectory(QueueDirectory);
        IsCoordinator = true;
      }
    }


    /// <summary>
    /// Adds request to run a game pair for a specified opening index.
    /// </summary>
    /// <param name="openingIndex"></param>
    public void EnqueueOpeningRequest(int openingIndex)
    {
      string fn = Path.Combine(QueueDirectory, openingIndex + "_pending");
//Console.WriteLine($"Enqueue {fn}");
      File.WriteAllText(fn, "");
    }

    void Dump()
    {
      Console.WriteLine("\r\nDIR " + DateTime.Now.Second + " " + DateTime.Now.Millisecond);
      foreach (string fileName in Directory.GetFiles(QueueDirectory, "*"))
      {
        Console.WriteLine("  " + fileName);
      }
    }

    static Random rand = new Random();

    /// <summary>
    /// Waits until a game request appears in the directory and then returns.
    /// </summary>
    /// <returns></returns>
    public int GetEnqueuedOpeningRequest()
    {
      while (true)
      {
        Thread.Sleep(100 + rand.Next(50));

        foreach (string fileName in Directory.GetFiles(QueueDirectory, "*_pending"))
        {
          FileInfo info = new FileInfo(fileName);
//Console.WriteLine("try get " + fileName);
//Dump();
          string renameFile = Path.Combine(QueueDirectory, fileName.Replace("_pending", "_processing"));
          try
          {
            int openingIndex = -1;

            // Try to open with lock to insure unique process.
            using (FileStream file = File.Open(fileName, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
            {
              if (!File.Exists(renameFile))
              {
                using (File.Create(renameFile)) ;
                string[] parts = info.Name.Split("_");
                openingIndex = int.Parse(parts[0]);
//Console.WriteLine("succeeded get " + fileName);
//Dump();

//Console.WriteLine("exists " + fileName + " " + File.Exists(fileName));
//Console.WriteLine("exists " + renameFile + " " + File.Exists(renameFile));
              }
            }
            File.Delete(fileName);
//Console.WriteLine("Post delete");
//Dump();
            return openingIndex;
          }
          catch
          {
            // Just ignore, another process grabbed the file.
            Thread.Sleep(rand.Next(50));
          }

        }
      }
    }

    /// <summary>
    /// Writes the results of a game pair.
    /// </summary>
    /// <param name="gameInfo"></param>
    /// <param name="gameReverseInfo"></param>
    public void WriteGamePairResult(TournamentGameInfo gameInfo, TournamentGameInfo gameReverseInfo)
    {
      string processingFN = Path.Combine(QueueDirectory, gameInfo.OpeningIndex + "_processing");
      string renameFile = processingFN.Replace("_processing", "_done");

      SysMisc.WriteObj(renameFile + ".1", gameInfo);
      SysMisc.WriteObj(renameFile + ".2", gameReverseInfo);
//Dump();
//Console.WriteLine("replace: " + processingFN + " to  " + renameFile);
      File.Move(processingFN, renameFile);
    }


    /// <summary>
    /// Retrieves the results of any single game pair which has completed
    /// </summary>
    /// <returns></returns>
    public (TournamentGameInfo gameInfo, TournamentGameInfo gameReverseInfo) GetGamePairResult()
    {
      while (true)
      {
        foreach (string fileName in Directory.GetFiles(QueueDirectory, "*_done"))
        {
          try
          {
            TournamentGameInfo g1 = null, g2 = null;
            string reportedFN;

            // Try to open with lock to insure unique process.
            using (FileStream file = File.Open(fileName, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
            {
              FileInfo info = new FileInfo(fileName);
              reportedFN = fileName.Replace("_done", "_reported");
              if (!File.Exists(reportedFN))
              {
                g1 = SysMisc.ReadObj<TournamentGameInfo>(fileName + ".1");
                g2 = SysMisc.ReadObj<TournamentGameInfo>(fileName + ".2");
              }
            }
            if (g1 != null)
            {
              File.Move(fileName, reportedFN);
              return (g1, g2);
            }

          }
          catch
          {
            Thread.Sleep(20 + rand.Next(20));
          }
        }
      }

      return (null, null);
    }

  }
}
