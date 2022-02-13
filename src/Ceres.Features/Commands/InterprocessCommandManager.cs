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

#endregion

namespace Ceres.Features.Commands
{
  /// <summary>
  /// Static manager of interprocess communication with running Ceres engines,
  /// allowing commands such as request to dump an analysis graph.
  /// 
  /// Commands are triggered via an interprocess named Event,
  /// and the specifics of the event (type, options) paseed via temporary file.
  /// </summary>
  public static class InterprocessCommandManager
  {
    /// <summary>
    /// Use a global named event.
    /// </summary>
    const string IPC_EVENT_NAME = "Global\\Ceres_InterprocessCommandManager";
    static string COMMAND_FILE_NAME() => Path.Combine(Path.GetTempPath(), "_CERES_COMMAND.txt");

    const float MIN_WAIT_SECONDS = 3; // to prevent runaway flood of events
    static DateTime lastEventTime = DateTime.Now;

    static EventWaitHandle ewh;


    /// <summary>
    /// Enqueues an interprocess command with specified options.
    /// </summary>
    /// <param name="command"></param>
    /// <param name="options"></param>
    public static void EnqueueCommand(string command, string options)
    {
      EventWaitHandle ewh = new(false, EventResetMode.AutoReset, IPC_EVENT_NAME, out bool createdNew);
      if (createdNew)
      {
        Console.WriteLine("No running instance of Ceres was found on this computer to queue pending command to.");
      }
      else
      {
        // Write details to file
        File.WriteAllLines(COMMAND_FILE_NAME(), new string[] { command, options });

        // Trigger event.
        ewh.Set();
      }

      ewh.Close();
    }


    /// <summary>
    /// Attempts to dequeue a pending command (if any).
    /// </summary>
    /// <returns></returns>
    public static (string command, string options) TryDequeuePendingCommand()
    {
      // Create event if not already extant (possibly from some other process).
      if (ewh == null)
      {
        ewh = new(false, EventResetMode.AutoReset, IPC_EVENT_NAME, out bool createdNew);
      }

      if ((DateTime.Now - lastEventTime).TotalSeconds < MIN_WAIT_SECONDS)
      {
        // Don't process events with excessive frequency to 
        // prevent runaway processing.
        return default;
      }

      if (ewh.WaitOne(0))
      {
        lastEventTime = DateTime.Now;

        // Read and process file with commands
        if (File.Exists(COMMAND_FILE_NAME()))
        {
          string[] lines = File.ReadAllLines(COMMAND_FILE_NAME());

          return (lines.Length > 0 ? lines[0] : null, 
                  lines.Length > 1 ? lines[1] : null);
        }
      }

      return default;
    }
  }

}
