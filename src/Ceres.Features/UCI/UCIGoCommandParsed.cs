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
using System.Diagnostics;
using Ceres.Base.Misc;
using Ceres.Chess;

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
    /// Remaining nodes for our side (proprietary UCI go mode)
    /// </summary>
    public readonly int? NodesOurs;

    /// <summary>
    /// Remaining nodes for opponent side (proprietary UCI go mode)
    /// </summary>
    public readonly int? NodesOpponent;

    /// <summary>
    /// Increment per move for our side (in milliseconds or nodes)
    /// </summary>
    public readonly int? IncrementOurs;

    /// <summary>
    /// Increment per move for opopnent (in milliseconds or nodes)
    /// </summary>
    public readonly int? IncrementOpponent;

    /// <summary>
    /// Number of moves left to go for the specified time allotment
    /// </summary>
    public readonly int? MovesToGo;

    /// <summary>
    /// Optionally a list of root-level moves to which the search is restricted.
    /// </summary>
    public readonly List<Move> SearchMoves;

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
          if (intToken == "")
          {
            return TakeIntToken();
          }

          // Allow numbers to include underscore characters (like in C# numeric literals)
          // for readability by stripping out these characters.
          intToken = intToken.Replace("_", "");

          if (!int.TryParse(intToken, out int ret))
          {
            throw new Exception($"Expected integer in go command instead saw {intToken}");
          }

          return ret;
        }

        switch (token)
        {
          case "infinite":
            Infinite = true;
            break;

          case "nodes":
            Nodes = Math.Max(1, TakeIntToken());
            break;

          case "movetime":
            MoveTime = Math.Max(1, TakeIntToken());
            break;

          case "movestogo":
            MovesToGo = TakeIntToken();

            // Convert big values to none,
            // as a workaround for clients that may use 
            // extremely large value as a proxy for "all."
            if (MovesToGo >= 99) MovesToGo = null;
            break;

          case "wtime":
            if (weAreWhite) 
              TimeOurs = Math.Max(1, TakeIntToken()); 
            else 
              TimeOpponent = Math.Max(1, TakeIntToken());
            break;

          case "btime":
            if (weAreBlack)
              TimeOurs = Math.Max(1, TakeIntToken());
            else
              TimeOpponent = Math.Max(1, TakeIntToken());
            break;

          case "wnodes":
            if (weAreWhite)
              NodesOurs = Math.Max(1, TakeIntToken());
            else
              NodesOpponent = Math.Max(1, TakeIntToken());
            break;

          case "bnodes":
            if (weAreBlack)
              NodesOurs = Math.Max(1, TakeIntToken());
            else
              NodesOpponent = Math.Max(1, TakeIntToken());
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

          case "depth":
            int depth = TakeIntToken();
            if (depth < 100)
            {
              Console.WriteLine($"Unsupported UCI go mode: {token}");
            }
            else
            {
              // Some clients may always send very large depths (like 9999).
              // Since these are effectively never binding, just ignore them.
            }
            break;

          case "moves":
          case "mate":
          case "ponder":
            Console.WriteLine($"Unsupported UCI go mode: {token}");
            IsValid = false;
            break;

          case "searchmoves":
            SearchMoves = new List<Move>();
            for (int i=partIndex; i<strParts.Length;i++)
            {
              SearchMoves.Add(Move.FromUCI(strParts[i]));
              partIndex++;
            }
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
