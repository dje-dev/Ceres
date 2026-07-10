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
using System.Linq;
using System.Collections.Generic;
using System.Globalization;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.Games.Utils;
using Ceres.Base.Misc;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.Features.Suites
{
  public delegate void EvaluatedPosCallback(EPDEntry epd, float correctnessScore, GameEngineSearchResult searchResult);


  /// <summary>
  /// Definition of one engine participating in a multiengine suite test.
  ///
  /// Wraps an EnginePlayerDef (whose underlying GameEngineDef may be a standard in-process
  /// Ceres engine or an external UCI/LC0 engine) together with a short display id, a flag
  /// indicating whether this is the baseline engine (against which the difference statistics
  /// are computed), and the search limit to use for this engine.
  /// </summary>
  public sealed class MultiEngineEntry
  {
    /// <summary>
    /// Short identifying string used as the column header in the live statistics block.
    /// </summary>
    public readonly string ID;

    /// <summary>
    /// Underlying engine player definition.
    /// </summary>
    public readonly EnginePlayerDef PlayerDef;

    /// <summary>
    /// If this engine is the baseline (the reference for the difference statistics).
    /// </summary>
    public readonly bool IsBaseline;

    /// <summary>
    /// Search limit used for this engine (may differ from limit of other engines).
    /// </summary>
    public readonly SearchLimit Limit;

    /// <summary>
    /// If the underlying engine is a standard in-process Ceres engine (MCTS or MCGS),
    /// for which the full set of statistics can be extracted.
    /// </summary>
    public bool IsCeresEngine => PlayerDef.EngineDef.IsCeresEngine;

    public MultiEngineEntry(string id, EnginePlayerDef playerDef, bool isBaseline, SearchLimit limit)
    {
      if (playerDef == null)
      {
        throw new ArgumentNullException(nameof(playerDef));
      }

      PlayerDef = playerDef;
      ID = id ?? playerDef.ID;
      IsBaseline = isBaseline;
      Limit = limit ?? playerDef.SearchLimit
              ?? throw new Exception($"No SearchLimit specified for engine {ID} (neither in the tuple nor on the EnginePlayerDef).");
    }

    public override string ToString() => $"<MultiEngineEntry {ID}{(IsBaseline ? "*" : "")} Limit={Limit} Def={PlayerDef.EngineDef}>";
  }

  /// <summary>
  /// Defines the parameters of a suite test.
  /// </summary>
  public class SuiteTestDef
  {
    /// <summary>
    /// Descriptive identifying string of suite test.
    /// </summary>
    public string ID;

    /// <summary>
    /// Target for logging messages.
    /// </summary>
    [NonSerialized] public TextWriter Output = Console.Out;

    /// <summary>
    /// Delegate to be called after each EPD has been evaluated.
    /// </summary>
    public EvaluatedPosCallback Callback;

    /// <summary>
    /// If extra EPD info should be dumped to Console at end of each position line (such as FEN).
    /// </summary>
    public bool DumpEPDInfo = false;

    /// <summary>
    /// Definition of the player engine for an external (UCI) engine (optional).
    /// </summary>
    public EnginePlayerDef ExternalEngineDef;

    /// <summary>
    /// Definition of the first Ceres player engine.
    /// </summary>
    public EnginePlayerDef CeresEngine1Def;

    /// <summary>
    /// Definition of the second ceres player engine (optional).
    /// </summary>
    public EnginePlayerDef CeresEngine2Def;

    /// <summary>
    /// If non-null, the suite is run in "multiengine mode": an arbitrary number of engines
    /// (each a standard in-process Ceres engine or an external engine) are compared at once,
    /// with the results shown in a single in-place-refreshed statistics block and returned as
    /// a comprehensive MultiEngineSuiteResult. In this mode the legacy CeresEngine1Def /
    /// CeresEngine2Def / ExternalEngineDef fields are not used; use SuiteTestRunner.RunMultiEngine.
    /// </summary>
    public List<MultiEngineEntry> MultiEngineDefs;

    /// <summary>
    /// If this definition is configured for multiengine mode.
    /// </summary>
    public bool IsMultiEngine => MultiEngineDefs != null;

    /// <summary>
    /// The (single) baseline engine entry in multiengine mode (null otherwise).
    /// </summary>
    public MultiEngineEntry BaselineEngine => MultiEngineDefs?.FirstOrDefault(e => e.IsBaseline);

    /// <summary>
    /// Name EPD file containing set of starting suite test positions to be used.
    /// The user setting "DirEPD" is consulted to determine the source directory.
    /// </summary>
    public string EPDFileName;

    /// <summary>
    /// If the line comes from a file in the Lichess puzzle format 
    /// (identifie, then FEN, then set of moves the first of which should be played before the puzzle).
    /// See: https://database.lichess.org/#puzzles.
    /// </summary>
    public bool EPDLichessPuzzleFormat;

    /// <summary>
    /// Optional filter predicate to select which raw lines in EPD are processed.
    /// </summary>
    public Predicate<string> EPDRawLineFilter;

    /// <summary>
    /// Optional filter predicate to select which entries in EPD are processed.
    /// </summary>
    public Predicate<EPDEntry> EPDFilter;

    /// <summary>
    /// If specified position should be accepted as part of the suite test.
    /// </summary>
    public Predicate<Position> AcceptPosPredicate;

    /// <summary>
    /// The number of first position in file to test (zero-based, defaults to first position).
    /// </summary>
    public int FirstTestPosition;


    /// <summary>
    /// Maximum number of suite positions to test.
    /// </summary>
    public int MaxNumPositions = int.MaxValue;

    /// <summary>
    /// Number of positions at beginning of file to be skipped.
    /// </summary>
    public int SkipNumPositions = 0;


    public bool RunCeres2Engine => CeresEngine2Def != null;
    public GameEngineDef Engine1Def => CeresEngine1Def.EngineDef;
    public GameEngineDef Engine2Def => CeresEngine2Def?.EngineDef;


    /// <summary>
    /// Constructor, specifying at least one Ceres engine
    /// and optionally a second Ceres engine and an 
    /// external engine.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="epdFileName"></param>
    /// <param name="ceresEngine1Def"></param>
    /// <param name="ceresEngine2Def"></param>
    /// <param name="externalEngineDef"></param>
    public SuiteTestDef(string id,
                        string epdFileName,
                        EnginePlayerDef ceresEngine1Def, 
                        EnginePlayerDef ceresEngine2Def = null,
                        EnginePlayerDef externalEngineDef = null)
    {
      if (ceresEngine2Def != null && !ceresEngine2Def.EngineDef.IsCeresEngine)
      {
        throw new Exception("ceresEngine2Def is expected to be for a Ceres engine");
      }

      if (externalEngineDef != null && externalEngineDef.EngineDef.IsCeresEngine)
      {
        throw new Exception("externalEngineDef is not expected to be for a Ceres engine");
      }

      ID = id;
      EPDFileName = epdFileName;

      CeresEngine1Def = ceresEngine1Def;
      CeresEngine2Def = ceresEngine2Def;
      ExternalEngineDef = externalEngineDef;
    }


    /// <summary>
    /// Constructor, specifying at least one Ceres engine
    /// and optionally a second Ceres engine and an 
    /// external engine.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="epdFileName"></param>
    /// <param name="ceresEngine1Def"></param>
    /// <param name="ceresEngine2Def"></param>
    /// <param name="externalEngineDef"></param>
    public SuiteTestDef(string id,
                        string epdFileName,
                        SearchLimit searchLimit,
                        GameEngineDef ceresEngine1Def,
                        GameEngineDef ceresEngine2Def = null,
                        GameEngineDef externalEngineDef = null)
      : this(id, epdFileName,
            new EnginePlayerDef(ceresEngine1Def, searchLimit),
            ceresEngine2Def == null ? null : new EnginePlayerDef(ceresEngine2Def, searchLimit),
            externalEngineDef == null ? null : new EnginePlayerDef(externalEngineDef, searchLimit))
    {

    }


    /// <summary>
    /// Constructor for "multiengine mode", comparing an arbitrary number of engines at once.
    ///
    /// Each engine is specified as a tuple of (short display id, engine player definition,
    /// is-baseline flag, search limit). The engine player definition may wrap either a standard
    /// in-process Ceres engine (GameEngineCeresInProcess / GameEngineCeresMCGSInProcess), for
    /// which the full set of statistics is available, or an external engine (UCI/LC0), for which
    /// only a subset of statistics is available. Exactly one engine should be marked as the
    /// baseline (against which the difference statistics are computed).
    ///
    /// The engineID and limit elements of a tuple may be null, in which case they fall back to
    /// the values on the EnginePlayerDef.
    ///
    /// Run this configuration with SuiteTestRunner.RunMultiEngine (not the legacy Run).
    /// </summary>
    /// <param name="id"></param>
    /// <param name="epdFileName"></param>
    /// <param name="engines"></param>
    public SuiteTestDef(string id,
                        string epdFileName,
                        params (string engineID, EnginePlayerDef enginePlayerDef, bool isBaseline, SearchLimit limit)[] engines)
    {
      if (engines == null || engines.Length == 0)
      {
        throw new ArgumentException("At least one engine must be specified for a multiengine suite test.", nameof(engines));
      }

      ID = id;
      EPDFileName = epdFileName;

      MultiEngineDefs = engines.Select(e => new MultiEngineEntry(e.engineID, e.enginePlayerDef, e.isBaseline, e.limit)).ToList();

      // Validate that there is exactly one baseline engine. If none is specified the first
      // engine is used as the baseline (with a warning); more than one is an error.
      int numBaseline = MultiEngineDefs.Count(e => e.IsBaseline);
      if (numBaseline > 1)
      {
        throw new Exception($"Exactly one engine may be marked as the baseline (found {numBaseline}).");
      }
      else if (numBaseline == 0)
      {
        Console.WriteLine($"WARNING: no baseline engine specified for multiengine suite '{id}'; using first engine ({MultiEngineDefs[0].ID}) as baseline.");
        MultiEngineEntry first = MultiEngineDefs[0];
        MultiEngineDefs[0] = new MultiEngineEntry(first.ID, first.PlayerDef, true, first.Limit);
      }

      // Distinct display ids keep the live block columns unambiguous.
      List<string> dupIDs = MultiEngineDefs.GroupBy(e => e.ID).Where(g => g.Count() > 1).Select(g => g.Key).ToList();
      if (dupIDs.Count > 0)
      {
        throw new Exception($"Engine ids in a multiengine suite test must be distinct (duplicated: {string.Join(", ", dupIDs)}).");
      }
    }


    /// <summary>
    /// Convenience factory producing a multiengine SuiteTestDef for a parameter sweep: the same
    /// (MCGS) engine and search limit are used for every engine, but each engine is an independent
    /// deep clone whose search/selection parameters are modified by one of the supplied parameter
    /// values. This makes it easy to sweep a range of values for a single parameter in one suite
    /// test by writing only the modifier, e.g.:
    ///
    ///   SuiteTestDef.CreateParameterSweep("sweep", epdFile, mcgsEngineDef, limit,
    ///       new[] { 0.01f, 0.02f, 0.04f, 0.08f, 0.12f },
    ///       (search, value) =&gt; search.VisitSuboptimalityRejectThreshold = value);
    ///
    /// One engine is created per value (with a short id formed from the value, e.g. "0.04"), and
    /// the middle engine is taken to be the baseline. The MCGS engine type is required so that the
    /// concrete ParamsSearch / ParamsSelect types are known to the modifiers.
    /// </summary>
    /// <param name="id">Suite test id.</param>
    /// <param name="epdFileName">EPD file of test positions.</param>
    /// <param name="engineDef">The MCGS engine definition to clone for each swept value.</param>
    /// <param name="searchLimit">Search limit applied to every engine.</param>
    /// <param name="paramValues">The parameter values to sweep (one engine each).</param>
    /// <param name="paramsSearchModifier">Optional action applied to each clone's ParamsSearch with its value.</param>
    /// <param name="paramsSelectModifier">Optional action applied to each clone's ParamsSelect with its value.</param>
    public static SuiteTestDef CreateParameterSweep(string id,
                                                    string epdFileName,
                                                    GameEngineDefCeresMCGS engineDef,
                                                    SearchLimit searchLimit,
                                                    float[] paramValues,
                                                    Action<ParamsSearch, float> paramsSearchModifier = null,
                                                    Action<ParamsSelect, float> paramsSelectModifier = null)
    {
      if (engineDef == null)
      {
        throw new ArgumentNullException(nameof(engineDef));
      }
      if (paramValues == null || paramValues.Length == 0)
      {
        throw new ArgumentException("At least one parameter value is required.", nameof(paramValues));
      }

      // Take the middle engine to be the baseline.
      int baselineIndex = paramValues.Length / 2;

      var engines = new (string engineID, EnginePlayerDef enginePlayerDef, bool isBaseline, SearchLimit limit)[paramValues.Length];
      HashSet<string> usedIDs = new HashSet<string>();

      for (int i = 0; i < paramValues.Length; i++)
      {
        float value = paramValues[i];

        // Each engine is an independent deep clone (so its parameters can be modified in isolation
        // and it has its own evaluator/search/select parameter objects).
        GameEngineDefCeresMCGS clone = ObjUtils.DeepClone(engineDef);
        paramsSearchModifier?.Invoke(clone.SearchParams, value);
        paramsSelectModifier?.Invoke(clone.SelectParams, value);

        // Build a short, unique id from the parameter value (e.g. "0.04").
        string baseID = value.ToString("0.####", CultureInfo.InvariantCulture);
        string engineID = baseID;
        int suffix = 2;
        while (!usedIDs.Add(engineID))
        {
          engineID = $"{baseID}#{suffix++}";
        }

        EnginePlayerDef playerDef = new EnginePlayerDef(clone, searchLimit, engineID);
        engines[i] = (engineID, playerDef, i == baselineIndex, searchLimit);
      }

      return new SuiteTestDef(id, epdFileName, engines);
    }


    /// <summary>
    /// Converts a legacy (one/two/three engine) definition into an equivalent multiengine-mode
    /// definition so the run can use the multiengine path and live statistics block. The first
    /// Ceres engine becomes the baseline; each engine keeps its own search limit, and the engines
    /// are labeled C1 / C2 / EX (matching the legacy console labels). No-op if already multiengine.
    /// </summary>
    public void ConfigureMultiEngineFromLegacy()
    {
      if (IsMultiEngine)
      {
        return;
      }

      if (CeresEngine1Def == null)
      {
        throw new Exception("Cannot enable multiengine mode: no engines are defined.");
      }

      List<MultiEngineEntry> list = new List<MultiEngineEntry>
      {
        // Engine 1 is the baseline.
        new MultiEngineEntry("C1", CeresEngine1Def, true, CeresEngine1Def.SearchLimit)
      };

      if (CeresEngine2Def != null)
      {
        list.Add(new MultiEngineEntry("C2", CeresEngine2Def, false, CeresEngine2Def.SearchLimit));
      }

      if (ExternalEngineDef != null)
      {
        list.Add(new MultiEngineEntry("EX", ExternalEngineDef, false, ExternalEngineDef.SearchLimit));
      }

      MultiEngineDefs = list;
    }


    public void DumpParams()
    {
      Console.WriteLine("SuiteTestDef.DumpParams DUMP - TODO");
#if NOT
      ParamsDump.DumpParams(Console.Out, true,
                 UCIEngine1Spec, UCIEngine2Spec,
                 EvaluatorDef1, EvaluatorDef2,
                 SearchLimitEngine1, SearchLimitEngine2,
                 SelectParams1, SelectParams2,
                 SearchParams1, SearchParams2,
                 OverrideTimeManager1, OverrideTimeManager2,
                 SearchParams1.Execution, SearchParams2.Execution
                 );
#endif
    }
  }
}
