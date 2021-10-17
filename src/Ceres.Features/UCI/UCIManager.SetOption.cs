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

using System.IO;
using System.Globalization;
using static System.FormattableString;

using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.Chess.UserSettings;
using Ceres.MCTS.Params;


#endregion

namespace Ceres.Features.UCI
{
  public partial class UCIManager
  {
    /// <summary>
    /// Optional text file to receive log of all UCI dialog between engine and client.
    /// </summary>
    string uciLogFileName = null;

    /// <summary>
    /// Optional text file to receive diagnostic log relating to Ceres engine moves.
    /// </summary>
    string searchLogFileName = null;

    /// <summary>
    /// If verbose move stats should be output at periodic intervals.
    /// </summary>
    bool logLiveStats = false;

    /// <summary>
    /// If detailed top-level move info should be output at end of move selection.
    /// </summary>
    bool verboseMoveStats = false;

    /// <summary>
    /// If the value head score should be output as a logistic (probability of winning),
    /// otherwise output as centipawn equivalent.
    /// </summary>
    bool scoreAsQ = false;

    /// <summary>
    /// If the W/D/L scores should each be shown in UCI info lines
    /// </summary>
    bool showWDL = false;

    /// <summary>
    /// If MultiPV info lines use N for child move.
    /// </summary>
    bool perPVCounters = false;

    /// <summary>
    /// If the experimental sibling evaluation feature is enabled.
    /// </summary>
    bool enableSiblingEval = false;

    /// <summary>
    /// Number of moves for which to output PVs in UCI input (multiPV mode is where numPV > 1).
    /// </summary>
    int numPV = 1;

    /// <summary>
    /// ParamsSelect.CPUCT
    /// </summary>
    float cpuct = new ParamsSelect().CPUCT;

    /// <summary>
    /// ParamsSelect.CPUCTBase
    /// </summary>
    float cpuctBase = new ParamsSelect().CPUCTBase;

    /// <summary>
    /// ParamsSelect.CPUCTFactor
    /// </summary>
    float cpuctFactor = new ParamsSelect().CPUCTFactor;

    /// <summary>
    /// ParamsSelect.CPUCTAtRoot
    /// </summary>
    float cpuctAtRoot = new ParamsSelect().CPUCTAtRoot;

    /// <summary>
    /// ParamsSelect.CPUCTBaseAtRoot
    /// </summary>
    float cpuctBaseAtRoot = new ParamsSelect().CPUCTBaseAtRoot;

    /// <summary>
    /// ParamsSelect.CPUCTFactorAtRoot
    /// </summary>
    float cpuctFactorAtRoot = new ParamsSelect().CPUCTFactorAtRoot;

    /// <summary>
    /// ParamsSelect.PolicySoftmax
    /// </summary>
    float policySoftmax = new ParamsSelect().PolicySoftmax;

    /// <summary>
    /// If futiltiy pruning (smart pruning) is disabled.
    /// </summary>
    bool futilityPruningDisabled = new ParamsSearch().FutilityPruningStopSearchEnabled == false;

    /// <summary>
    /// Amount of seconds subtracted from time limits to compensate for overhead/latency.
    /// </summary>
    float moveOverheadSeconds = new ParamsSearch().MoveOverheadSeconds;

    /// <summary>
    /// Maximum size of tree allowed (in visits).
    /// </summary>
    int? maxTreeVisits = null;

    /// <summary>
    /// Maximum size of tree allowed (in nodes).
    /// </summary>
    int? maxTreeNodes = CeresUserSettingsManager.Settings.MaxTreeNodes;

    /// <summary>
    /// Multiplier applied to all requested time limits.
    /// </summary>
    float searchLimitMultiplier = 1.0f;

    /// <summary>
    /// First play urgency coefficient.
    /// </summary>
    float fpu = new ParamsSelect().FPUValue;

    /// <summary>
    /// First play urgency (at root) coefficient.
    /// </summary>
    float fpuAtRoot = new ParamsSelect().FPUValueAtRoot;


    void ProcessSetOption(string command)
    {
      string[] parts = command.Split(" ");

      if (parts.Length < 4
       || parts[0].ToLower() != "setoption"
       || parts[1].ToLower() != "name"
       || parts[3].ToLower() != "value"
       )
      {
        OutStream.WriteLine("Expected command of form setoption name <option_name> value <option_value>");
        return;
      }

      string name = parts[2];
      string value = command.Substring(command.IndexOf("value ") + 6);

      switch (name.ToLower())
      {
        case "weightsfile":
          ProcessWeightsFile(value);
          break;

        case "searchlogfile":
          searchLogFileName = value == "" ? null : value;
          if (CeresEngine != null) CeresEngine.SearchLogFileName = searchLogFileName;
          break;

        case "logfile":
          uciLogFileName = value == "" ? null : value;
          if (uciLogFileName == null)
          {
            uciLogWriter = null;
          }
          else
          {
            uciLogWriter = new StreamWriter(new FileStream(uciLogFileName, FileMode.Append, FileAccess.Write));
          }
          break;

        case "loglivestats":
          SetBool(value, ref logLiveStats);
          break;

        case "maxtreevisits":
          SetInt(value, 1, int.MaxValue, ref maxTreeVisits);
          break;

        case "maxtreenodes":
          SetInt(value, 1, int.MaxValue, ref maxTreeNodes);
          break;

        case "searchlimitmultiplier":
          SetFloat(value, 0.001f, 1_000_000, ref searchLimitMultiplier);
          break;

        case "moveoverheadms":
          float moveOverheadMilliseconds = 0;
          SetFloat(value, 0, int.MaxValue, ref moveOverheadMilliseconds);
          moveOverheadSeconds = moveOverheadMilliseconds / 1000f;
          break;

        case "smartpruningfactor":
          float factor = -1;
          SetFloat(value, 0, int.MaxValue, ref factor);
          if (factor == 0)
            futilityPruningDisabled = true;
          else if (factor == ParamsSearch.LC0_DEFAULT_SMART_PRUNING_FACTOR)
            futilityPruningDisabled = false;
          else
            OutStream.Write($"Ceres does not support {factor}, only value 0 (indicating turned off) or 1.33 (indicating default) for SmartPruningFactor");
          break;

        case "verbosemovestats":
          SetBool(value, ref verboseMoveStats);
          break;

        case "scoretype":
          value = value.ToLower();
          if (value == "centipawn")
            scoreAsQ = false;
          else if (value == "w-l" || value == "q")
            scoreAsQ = true;
          else
            OutStream.Write("Invalid value for ScoreType, allowable values are Centipawn, Q or W-L");
          break;

        //option name ScoreType type combo default centipawn var centipawn var Q var W-L

        case "multipv":
          SetInt(value, 1, int.MaxValue, ref numPV);
          break;

        case "perpvcounters":
          SetBool(value, ref perPVCounters);
          break;

        case "enablesiblingeval":
          SetBool(value, ref enableSiblingEval);
          break;          

        case "uci_showwdl":
          SetBool(value, ref showWDL);
          break;

        case "reducedmemorymode":
          bool reducedMemory = false;
          SetBool(value, ref reducedMemory);
          CeresUserSettingsManager.Settings.ReducedMemoryMode = reducedMemory;
          break;          

        case "syzygypath":
          if (!Directory.Exists(value))
          {
            OutStream.WriteLine($"Path not found: { value }");
          }
          else
          {
            CeresUserSettingsManager.Settings.SyzygyPath = value;
            CeresUserSettingsManager.Settings.DirTablebases = null;
          }
          break;


        case "cpuct":
          SetFloat(value, 0, float.MaxValue, ref cpuct);
          break;

        case "cpuctbase":
          SetFloat(value, 1, float.MaxValue, ref cpuctBase);
          break;

        case "cpuctfactor":
          SetFloat(value, 0, float.MaxValue, ref cpuctFactor);
          break;

        case "cpuctatroot":
          SetFloat(value, 0, float.MaxValue, ref cpuctAtRoot);
          break;

        case "cpuctbaseatroot":
          SetFloat(value, 1, float.MaxValue, ref cpuctBaseAtRoot);
          break;

        case "cpuctfactoratroot":
          SetFloat(value, 0, float.MaxValue, ref cpuctFactorAtRoot);
          break;

        case "policytemperature":
          SetFloat(value, 0.1f, float.MaxValue, ref policySoftmax);
          break;

        case "fpu":
          SetFloat(value, 0, float.MaxValue, ref fpu);
          break;

        case "fpuatroot":
          SetFloat(value, 0, float.MaxValue, ref fpuAtRoot);
          break;
      }
    }


    private void ProcessWeightsFile(string value)
    {
      if (taskSearchCurrentlyExecuting != null)
      {
        OutStream.WriteLine("uci info Implementation limitation: cannot change weights while search is running.");
      }
      else
      {
        if (value == null || value == "")
        {
          OutStream.WriteLine("uci info Network file name expected");
        }
        else
        {
          if (EvaluatorDef != null && EvaluatorDef.Nets[0].Net.NetworkID.ToLower() == value.ToLower())
          {
            OutStream.WriteLine($"uci info Specified network file is already {value}");
          }
          else
          {
            ReinitializeEngine();
            NetworkSpec = new NNNetSpecificationString(value);
            CreateEvaluator();
          }
        }
      }
    }

    void SetBool(string boolStr, ref bool value)
    {
      if (boolStr.ToLower() == "true")
      {
        value = true;
      }
      else if (boolStr.ToLower() == "false")
      {
        value = false;
      }
      else
      {
        OutStream.WriteLine("Invalid value, expected true or false");
      }
    }


    void SetInt(string intStr, int minValue, int maxValue, ref int? value)
    {
      int setValue = 0;
      SetInt(intStr, minValue, maxValue, ref setValue);
      value = setValue;
    }

    void SetInt(string intStr, int minValue, int maxValue, ref int value)
    {
      if (!int.TryParse(intStr, out int newValue))
      {
        OutStream.WriteLine("Invalid value, expected integer");
      }

      if (newValue < minValue)
      {
        OutStream.WriteLine($"Value below minimum of {minValue}");
      }
      else if (newValue > maxValue)
      {
        OutStream.WriteLine($"Value above maximum of {maxValue}");
      }
      else
      {
        value = newValue;
      }
    }

    void SetFloat(string floatStr, float minValue, float maxValue, ref float value)
    {
      if (!float.TryParse(floatStr, NumberStyles.Any, CultureInfo.InvariantCulture, out float newValue))
      {
        OutStream.WriteLine("Invalid value, expected number");
      }

      if (newValue < minValue)
      {
        OutStream.WriteLine($"Value below minimum of {minValue}");
      }
      else if (newValue > maxValue)
      {
        OutStream.WriteLine($"Value above maximum of {maxValue}");
      }
      else
        value = newValue;
    }

    static string SetOptionUCIDescriptions => Invariant(
@$"
option name WeightsFile type string default <from DefaultNetworkSpecString in Ceres.json>
option name LogFile type string default
option name SearchLogFile type string default
option name MultiPV type spin default 1 min 1 max 500
option name VerboseMoveStats type check default false
option name LogLiveStats type check default false
option name SmartPruningFactor type string default 1.33
option name MoveOverheadMs type spin default {new ParamsSearch().MoveOverheadSeconds * 1000} min 0 max 100000000
option name PerPVCounters type check default false
option name ScoreType type combo default centipawn var centipawn var Q var W-L
option name UCI_ShowWDL type check default false
option name SyzygyPath type string default
option name CPUCT type string default {new ParamsSelect().CPUCT}
option name CPUCTAtRoot type string default {new ParamsSelect().CPUCTAtRoot}
option name CPuctBase type string default {new ParamsSelect().CPUCTBase}
option name CPuctBaseAtRoot type string default {new ParamsSelect().CPUCTBaseAtRoot}
option name CPuctFactor type string default {new ParamsSelect().CPUCTFactor}
option name CPuctFactorAtRoot type string default {new ParamsSelect().CPUCTFactorAtRoot}
option name PolicyTemperature type string default {new ParamsSelect().PolicySoftmax}
option name FPU type string default {new ParamsSelect().FPUValue}
option name FPUAtRoot type string default {new ParamsSelect().FPUValueAtRoot}
option name SearchLimitMultiplier type string default 1.00
option name MaxTreeVisits type string default
option name MaxTreeNodes type string default
option name ReducedMemoryMode type check default false
option name EnableSiblingEval type check default false
");
    /* 
option name ConfigFile type string default lc0.config
option name HistoryFill type combo default fen_only var no var fen_only var always
";
*/

  }

}