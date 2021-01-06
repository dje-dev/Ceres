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
using System.Diagnostics;
using Ceres.Base.Misc;

#endregion

namespace Ceres.Features.UCI
{
  /// <summary>
  /// Manages parsing of the various forms of the UCI command "go."
  /// </summary>
  internal record UCIGoCommandParsed
  {
    /// <summary>
    /// If the go command was successfully parsed.
    /// </summary>
    public readonly bool IsValid;

    /// <summary>
    /// Number of nodes to process
    /// </summary>
    public readonly int? Nodes;

    /// <summary>
    /// If inifinite analysis was requested
    /// </summary>
    public readonly bool Infinite = false;

    /// <summary>
    /// Requested move time (in milliseconds)
    /// </summary>
    public readonly int? MoveTime;

    /// <summary>
    /// Remaining time for our side (in milliseconds)
    /// </summary>
    public readonly int? TimeOurs;

    /// <summary>
    /// Remaining time for opponent side (in milliseconds)
    /// </summary>
    public readonly int? TimeOpponent;

    /// <summary>
    /// Increment time per move for our side (in milliseconds)
    /// </summary>
    public readonly int? IncrementOurs;

    /// <summary>
    /// Increment time per move for opopnent (in milliseconds)
    /// </summary>
    public readonly int? IncrementOpponent;

    /// <summary>
    /// Number of moves left to go for the specified time allotment
    /// </summary>
    public readonly int? MovesToGo;


    /// <summary>
    /// Constructor from UCI string received
    /// </summary>
    /// <param name="goCommand"></param>
    /// <param name="weAreWhite"></param>
    public UCIGoCommandParsed(string goCommand, bool weAreWhite)
    {
      bool weAreBlack = !weAreWhite;

      // Remove any extraneous whitespaces
      goCommand = StringUtils.WhitespaceRemoved(goCommand);

      string[] strParts = goCommand.Split(" ");
      Debug.Assert(strParts[0] == "go");

      int partIndex = 1;

      IsValid = true;

      while (partIndex < strParts.Length)
      {
        string token = strParts[partIndex++];

        int TakeIntToken()
        {
          string intToken = strParts[partIndex++];

          // If we see an empty string (due to extraneous spaces)
          if (intToken == "") return TakeIntToken();

          // Allow numbers to include underscore characters (like in C# numeric literals)
          // for readability by stripping out these characters.
          intToken = intToken.Replace("_", "");

          if (!int.TryParse(intToken, out int ret))
            throw new Exception("Expected integer in go command instead saw {intToken}");
          return ret;
        }

        switch (token)
        {
          case "infinite":
            Infinite = true;
            break;

          case "nodes":
            Nodes = TakeIntToken();
            break;

          case "movetime":
            MoveTime = TakeIntToken();
            break;

          case "movestogo":
            MovesToGo = TakeIntToken();
            break;

          case "wtime":
            if (weAreWhite) 
              TimeOurs = Math.Max(0, TakeIntToken()); 
            else 
              TimeOpponent = Math.Max(0, TakeIntToken());
            break;

          case "btime":
            if (weAreBlack)
              TimeOurs = Math.Max(0, TakeIntToken());
            else
              TimeOpponent = Math.Max(0, TakeIntToken());
            break;

          case "winc":
            if (weAreWhite) 
              IncrementOurs = TakeIntToken(); 
            else 
              IncrementOpponent = TakeIntToken();
            break;

          case "binc":
            if (weAreBlack)
              IncrementOurs = TakeIntToken();
            else
              IncrementOpponent = TakeIntToken();
            break;

          case "moves":
          case "mate":
          case "depth":
          case "ponder":
          case "searchmoves":
            Console.WriteLine($"Unsupported UCI go mode: {token}");
            IsValid = false;
            break;

          default:
            Console.WriteLine($"Unexpected UCI token within go command: {token}");
            IsValid = false;
            break;

        }
      }

    }
  }
}
