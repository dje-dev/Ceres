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
using System.Text;

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Static helper methods for working with the system Console.
  /// </summary>
  public static class ConsoleUtils
  {
    static bool? colorDisabled = null;

    /// <summary>
    /// Checks if NO_COLOR is set in the environment, per https://no-color.org. 
    /// </summary>
    /// <returns></returns>
    private static bool IsNoColorEnabled()
    {
      if (colorDisabled is not null)
      {
        return colorDisabled.Value;
      }
      else
      {
        string noColor = System.Environment.GetEnvironmentVariable("NO_COLOR");
        colorDisabled = !string.IsNullOrEmpty(noColor);
        return colorDisabled.Value;
      }
    }


    /// <summary>
    /// Writes a line of output to Console in specified color, unless NO_COLOR is set.
    /// </summary>
    /// <param name="str"></param>
    /// <param name="color"></param>
    /// <param name="endLine"></param>
    public static void WriteLineColored(ConsoleColor color, string str, bool endLine = true)
    {
      if (IsNoColorEnabled())
      {
        Console.Write(str);
        if (endLine)
        {
          Console.WriteLine();
        }
        return;
      }
      ConsoleColor priorColor = Console.ForegroundColor;
      Console.ForegroundColor = color;
      Console.Write(str);
      Console.ForegroundColor = priorColor;

      if (endLine)
      {
        Console.WriteLine();
      }
    }

    /// <summary>
    /// Prompts for and returns a string from the Console, without echoing the input.
    /// </summary>
    /// <param name="prompt"></param>
    /// <returns></returns>
    public static string ConsoleReadStringHidden(string prompt)
    {
      Console.Write(prompt + ": ");
      StringBuilder inputStr = new StringBuilder();
      while (true)
      {
        ConsoleKeyInfo keyInfo = Console.ReadKey(intercept: true);
        if (keyInfo.Key == ConsoleKey.Enter)
        {
          Console.WriteLine();
          return inputStr.ToString();
        }
        else if (keyInfo.Key == ConsoleKey.Backspace)
        {
          if (inputStr.Length > 0)
          {
            inputStr.Remove(inputStr.Length - 1, 1);

            // Move cursor one step back, write a space to erase the last dot, and then move one step back again.
            Console.SetCursorPosition(Console.CursorLeft - 1, Console.CursorTop);
            Console.Write(" ");
            Console.SetCursorPosition(Console.CursorLeft - 1, Console.CursorTop);
          }
        }
        else
        {
          inputStr.Append(keyInfo.KeyChar);
          Console.Write(".");
        }
      }
    }

    /// <summary>
    /// Invokes the specified action, suppressing all Console output.
    /// </summary>
    /// <param name="action"></param>
    public static void InvokeNoConsoleOutput(Action action)
    {
      TextWriter originalConsoleOut = Console.Out;
      Console.SetOut(TextWriter.Null);
      try
      {
        action();
      }
      finally
      {
        Console.SetOut(originalConsoleOut);
      }
    }
  }
}
