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
using Ceres.Base.Misc;
using Ceres.Chess.NNEvaluators.Specifications;

#endregion

namespace Ceres.Chess.UserSettings
{
  /// <summary>
  /// Static helper method which interacts with user
  /// at the Console to provide a minimal initialization 
  /// of the Ceres settings file.
  /// </summary>
  public static partial class CeresUserSettingsManager
  {
    /// <summary>
    /// Prompts user to enter certain user settings via Console
    /// and writes back to Ceres.json.
    /// </summary>
    public static void DoSetupInitialize()   
    {
      string dirLC0Networks = GetDirectoryString("  LC0 network weights files (optional)  : ", false);
      string dirLC0Binaries = GetDirectoryString("  LC0 binaries (optional)               : ", false);
      string defaultNetworkString = GetString("  Default network specification (e.g. \"LC0:703810\")                    : ", 
                                              s => NNNetSpecificationString.IsValid(s));
      string defaultDeviceString = GetString("  Default device specification (e.g. \"GPU:0\")                          : ", 
                                             s => NNDevicesSpecificationString.IsValid(s));

      settings = new CeresUserSettings();
      settings.DirLC0Binaries = dirLC0Binaries;
      settings.DirLC0Networks = dirLC0Networks;
      settings.DefaultNetworkSpecString = defaultNetworkString;
      settings.DefaultDeviceSpecString = defaultDeviceString;
      SaveToDefaultFile();
      LoadFromDefaultFile(); // this cause directory registration
      
      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Cyan, "NEW CONTENTS OF " + CeresUserSettingsManager.DefaultCeresConfigFileName);
      Console.WriteLine("(additional changes can be made by editing in a text editor or using SETOP command in Ceres");

      string fileContext = File.ReadAllText(CeresUserSettingsManager.DefaultCeresConfigFileName);
      Console.WriteLine(fileContext);


    }

    static string GetString(string description, Predicate<string> validation = null)
    {
      string str = null;
      while (true)
      {
        Console.Write(description);
        str = Console.ReadLine();
        Console.WriteLine();
        if (validation != null && !validation.Invoke(str))
          Console.WriteLine("Invalid value");
        else
          return str;
      }
    }

    static string GetDirectoryString(string description, bool required)
    {
      string dir = null;
      while (true)
      {
        Console.Write(StringUtils.Sized("  Enter path to directory with " + description, 73));
        dir = Console.ReadLine();
        if (dir.EndsWith("\\")) dir = dir.Substring(dir.Length - 1);

        Console.WriteLine();
        if (dir != "" && !Directory.Exists(dir))
        {
          Console.WriteLine("No such directory.");
        }
        else if (dir == "" && required)
        {
          Console.WriteLine("This directory is required.");
        }
        else
        {
          return dir;
        }
      }
    }

  }

}
