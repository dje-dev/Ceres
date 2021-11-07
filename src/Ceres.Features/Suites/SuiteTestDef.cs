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
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;

#endregion

namespace Ceres.Features.Suites
{
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
    public Predicate<string> EPDFilter;

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
    public GameEngineDefCeres Engine1Def => CeresEngine1Def.EngineDef as GameEngineDefCeres;
    public GameEngineDefCeres Engine2Def => CeresEngine2Def?.EngineDef as GameEngineDefCeres;


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
      if (ceresEngine2Def != null && (ceresEngine2Def.EngineDef is not GameEngineDefCeres))
      {
        throw new Exception("ceresEngine2Def is expected to be for a Ceres engine");
      }

      if (externalEngineDef != null && (externalEngineDef.EngineDef is GameEngineDefCeres))
      {
        throw new Exception("externalEngineDef is not expected to be for a Ceres engine, instead GameEngineDefLC0 or GameEngineDefUCI");
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
