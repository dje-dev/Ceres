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
using System.Collections.Generic;
using System.Linq;

#endregion

namespace Ceres.Train
{
  /// <summary>
  /// Manages logging of output to a file and/or console.
  /// </summary>
  public class FileLogger
  {
    // The name of the file to which output is logged.
    public readonly string LiveLogFileName;

    /// <summary>
    /// If true, output is also sent to the console.
    /// </summary>
    public readonly bool Verbose;

    /// <summary>
    /// Number of lines output to the log file.
    /// </summary>
    public int NumLinesOutput { private set; get; } = 0;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="logdir"></param>
    /// <param name="liveLogFileName"></param>
    /// <param name="verbose"></param>
    /// <param name="fileHeaderString"></param>
    public FileLogger(string logDir, string liveLogFileName = null, bool verbose = false, string fileHeaderString = null)
    {
      LiveLogFileName = Path.Combine(logDir, liveLogFileName ?? Path.GetRandomFileName() + ".txt");
      Verbose = verbose;

      if (fileHeaderString != null)
      {
        AddLine(liveLogFileName + Environment.NewLine + Environment.NewLine, true);
      }
    }


    /// <summary>
    /// Adds a line to the log file and optionally to the console.
    /// </summary>
    /// <param name="line"></param>
    public void AddLine(string line, bool isPrefix = false)
    {
      string outLine = DateTime.Now + " " + line;
      File.AppendAllText(LiveLogFileName, outLine + Environment.NewLine);

      if (!isPrefix)
      {
        NumLinesOutput++;
      }

      if (Verbose)
      {
        Console.WriteLine(line);
      }
    }


    #region Info string loading

    /// <summary>
    /// Log files may contain info lines (starting with INFO:) 
    /// that contain key/value pairs intended for extraction.
    /// </summary>
    Dictionary<string, string> logLinesStartingWithINFO;


    /// <summary>
    /// Builds a dictionary of INFO lines from the log file.
    /// </summary>
    public void LoadInfoDictionary()
    {
      // Lines of the form: date time INFO: key value
      string[] logLines = File.ReadAllLines(LiveLogFileName);
      logLinesStartingWithINFO = logLines.Where(line => line.Contains("INFO:"))
        .ToDictionary(line => line.Split(" ")[4], line => line.Split(" ")[^1]);
    }


    /// <summary>
    /// Gets an int value from the INFO lines of the log file.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public long GetInfoLong(string key)
    {
      if (!logLinesStartingWithINFO.ContainsKey(key))
      {
        throw new Exception(key + " not found in log file");
      }

      if (!long.TryParse(logLinesStartingWithINFO[key], out long intValue))
      {
        throw new Exception($"Could not parse int value from {logLinesStartingWithINFO[key]} for " + key);
      }

      return intValue;
    }


    /// <summary>
    /// Gets a string value from the INFO lines of the log file.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public string GetInfoStr(string key)
    {
      if (!logLinesStartingWithINFO.ContainsKey(key))
      {
        return null;
      }

      return logLinesStartingWithINFO[key];
    }

    #endregion

  }
}
