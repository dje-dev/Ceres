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
using Ceres.Base.Misc;

#endregion

namespace Ceres.Commands
{
  internal static class HelpCommands
  {
    internal const string VALID_COMMANDS = "HELP, UCI, ANALYZE, SUITE, TOURN, SYSBENCH, BACKENDBENCH, BENCHMARK, SETOPT or SETUP";

    internal static void ProcessHelpCommand(string cmd)
    {
      cmd = cmd.ToUpper();
      string[] parts = cmd.Split(" ");
      if (parts.Length != 2)
      {
        Console.WriteLine();
        ConsoleUtils.WriteLineColored(ConsoleColor.Cyan, "AVAILABLE SUBCOMMANDS");
        DumpAllHelp();
        System.Environment.Exit(0);
      }
      else if (parts[1] == "SETUP")
        DumpHelpText(CERES_HELP_SETUP);
      else if (parts[1] == "UCI")
        DumpHelpText(CERES_HELP_UCI);
      else if (parts[1] == "SUITE")
        DumpHelpText(CERES_HELP_SUITE);
      else if (parts[1] == "TOURN")
        DumpHelpText(CERES_HELP_TOURN);
      else if (parts[1] == "ANALYZE")
        DumpHelpText(CERES_HELP_ANALYZE);
      else if (parts[1] == "SETOPT")
        DumpHelpText(CERES_HELP_SETOPT);
      else if (parts[1] == "SYSBENCH")
        DumpHelpText(CERES_HELP_SYSBENCH);
      else if (parts[1] == "BACKENDBENCH")
        DumpHelpText(CERES_HELP_BACKENDBENCH);
      else if (parts[1] == "BENCHMARK")
        DumpHelpText(CERES_HELP_BENCHMARK);
      else
        DispatchCommands.ShowErrorExit($"Unrecognized command {parts[1]}, try " + HelpCommands.VALID_COMMANDS);
      System.Environment.Exit(0);
    }

    static void DumpHelpText(string text)
    {
      int index = 0;
      foreach (string line in text.Split("\n"))
      {
        if (index++ == 0)
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, line);
        else
          Console.WriteLine(line);
      }
    }

    const string CERES_HELP_SETUP =
    @"  SETUP - Prompts for user settings and initializes a new Ceres.json configuration file with these values.
";

    const string CERES_HELP_UCI =
    @"  UCI - Run in UCI Engine Mode (equivalently omit this subcommand)
    Optional key/values   :  { network, device, pruning }
";

    const string CERES_HELP_ANALYZE =
    @"  ANALYZE - Run search on specified FEN position and dump detailed move statistics and principal variation at end of search.
    Required final arg    : a FEN to be analyzed, or the keyword startpos
    Optional key/values   : { network, device, limit, opponent, network-opponent, device-opponent, limit-opponent, pruning }
    Example               : Ceres ANALYZE limit=30sm network=LC0:73810 device=GPU:0 opponent=lc0 startpos
";

    const string CERES_HELP_SUITE =
    @"  SUITE -  Run search on all positions in a suite of test positions from an EPD file and quantify agreement with specified correct moves.
    Required key/values   :  epd
    Optional key/values   :  { network, device, limit, opponent, network-opponent, device-opponent, limit-opponent, pruning }
    Examples              :  Ceres SUITE epd=sts.epd limit=1000nm network=LC0:73810 device=GPU:0
";

    const string CERES_HELP_TOURN =
    @"  TOURN - Run a tournament of games between Ceres and (possibly) another external UCI chess engine.
    Optional key/values   :  { openings, show-moves, network, device, limit, opponent, network-opponent, device-opponent, pruning}
    Example               :  Ceres TOURN opponent=stockfish11.exe limit=1000nm limit_opponent=875_000nm openings=drawkiller.pgn showmoves=false
";

    const string CERES_HELP_SETOPT =
    @"  SETOPT - Modify a specified option in the Ceres configuration file with a given value.
    Optional key/ values  :  { network, device, limit, limit_opponent, epd, openings, opponent, lc0_files }
    Example               :  sCeres SETOPT network = LC0:73810 device=GPU:0 lc0_files=c:\lc0_weight_files
";

    const string CERES_HELP_SYSBENCH =
    @"  SYSBENCH - Runs CPU and GPU benchmarks and dumps summary result to console.
";

    const string CERES_HELP_BACKENDBENCH =
    @"  BACKENDBENCH - Runs neural network backend benchmark.
    Optional key/values   : { network, device }
    Example               : Ceres BACKENDBENCH network=LC0:73810 device=GPU:0

";

    const string CERES_HELP_BENCHMARK =
    @"  BENCHMARK - Runs search benchmarks across a set of standard benchmark positinos.
    Optional key/values   : { network, device, limit, opponent, network-opponent, device-opponent, limit-opponent, pruning }
    Example               : Ceres BENCHMARK limit=30sm network=LC0:73810 device=GPU:0 opponent=lc0
";


    static void DumpAllHelp()
    {
      DumpHelpText(CERES_HELP_UCI);
      DumpHelpText(CERES_HELP_ANALYZE);
      DumpHelpText(CERES_HELP_TOURN);
      DumpHelpText(CERES_HELP_SUITE);
      DumpHelpText(CERES_HELP_SYSBENCH);
      DumpHelpText(CERES_HELP_BACKENDBENCH);
      DumpHelpText(CERES_HELP_BENCHMARK);
      DumpHelpText(CERES_HELP_SETOPT);
      DumpHelpText(CERES_HELP_SETUP);
    }


  }
}
