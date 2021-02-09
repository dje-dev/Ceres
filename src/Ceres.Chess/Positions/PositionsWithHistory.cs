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
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;


using Ceres.Chess.Games.Utils;
using Ceres.Chess.MoveGen;
using Ceres.Chess.UserSettings;
using Ceres.Chess.Textual.PgnFileTools;

#endregion

namespace Ceres.Chess.Positions
{
  /// <summary>
  /// Represents an enumerable and indexable set of MoveSequence 
  /// sourced from somewhere (such as a PGN or EPD file).
  /// 
  /// TODO: it would be more elegant to use polymorphism as a way 
  ///       of supporting the various modes (PGN, EPD, etc.).
  /// </summary>
  public class PositionsWithHistory : IEnumerable<PositionWithHistory>
  {
    #region Private data

    List<PGNGame> openings;
    List<EPDEntry> epds = null;
    List<string> fensAndMoves = null;
    List<PositionWithHistory> moveSequences = null;

    #endregion

    #region Static factory methods

    /// <summary>
    /// Returns a position source that returns positions from one or more MoveSequences.
    /// </summary>
    /// <param name="fen"></param>
    /// <returns></returns>
    public static PositionsWithHistory FromMoveSequencess(params PositionWithHistory[] moveSequences)
    {
      PositionsWithHistory source = new PositionsWithHistory();
      source.moveSequences = new List<PositionWithHistory>();
      foreach (PositionWithHistory moveSequence in moveSequences)
        source.moveSequences.Add(moveSequence);
      return source;
    }

    /// <summary>
    /// Returns a position source that returns positions from one or more FENs.
    /// </summary>
    /// <param name="fen"></param>
    /// <returns></returns>
    public static PositionsWithHistory FromFENs(params string[] fens)
    {
      PositionsWithHistory source = new PositionsWithHistory();
      source.fensAndMoves = new List<string>();
      foreach (string fen in fens)
        source.fensAndMoves.Add(fen);
      return source;
    }

    /// <summary>
    /// Returns a position source that returns a single starting position (possibly multiple times).
    /// </summary>
    /// <param name="fen"></param>
    /// <returns></returns>
    public static PositionsWithHistory FromFEN(string fen, int repeatCount)
    {
      if (repeatCount > 1_000_000) throw new ArgumentOutOfRangeException(nameof(repeatCount), "is too large, maximum 1_000_000");

      PositionsWithHistory source = new PositionsWithHistory();
      source.fensAndMoves = new List<string>();
      for (int i=0; i<repeatCount;i++)
        source.fensAndMoves.Add(fen);
      return source;
    }


    public static PositionsWithHistory FromEPDOrPGNFile(string fileName)
    {
      PositionsWithHistory source = new PositionsWithHistory();
      source.LoadOpenings(fileName);
      return source;
    }

    #endregion


    /// <summary>
    /// Returns the number of start positions that this source provides.
    /// </summary>
    public int Count
    {
      get
      {
        if (epds != null)
          return epds.Count;
        else if (fensAndMoves != null)
          return fensAndMoves.Count;
        else if (openings != null)
          return openings.Count;
        else if (moveSequences != null)
          return moveSequences.Count;
        else
          throw new NotImplementedException();
      }
    }


    /// <summary>
    /// Internal worker method to return an enumerator over all PositionsWithHistory.
    /// </summary>
    private IEnumerable<PositionWithHistory> MoveSequences
    {
      get
      {
        for (int i = 0; i < Count; i++)
          yield return GetAtIndex(i);
        yield break;
      }
    }


    /// <summary>
    /// Indexer that returns the position at a specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public PositionWithHistory this[int index] => GetAtIndex(index);


    /// <summary>
    /// Returns the position at a specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public PositionWithHistory GetAtIndex(int index)
    {
      if (moveSequences != null)
      {
        return moveSequences[index];
      }
      if (fensAndMoves != null)
      {
        string fenAndMoves = fensAndMoves[index];
        string[] parts = fenAndMoves.Split(" ");
        string fen = null;
        string moves = null;
        if (fenAndMoves.Contains("moves"))
        {
          int indexMoves = fenAndMoves.IndexOf("moves");
          fen = fenAndMoves.Substring(0, indexMoves);
          moves = fenAndMoves.Substring(indexMoves + 6);
        }
        else
          fen = fenAndMoves;
        return PositionWithHistory.FromFENAndMovesUCI(fen, moves);
      }
      else if (openings != null)
      {
        PGNGame game = openings[index];
        return game.Moves;
      }
      else if (epds != null)
      {
        return new PositionWithHistory(MGPosition.FromFEN(epds[index].FEN));
      }
      else
        throw new Exception("PositionsWithHistory exhausted set of game start positions.");
    }

  
    public void LoadOpeningsFENsAndMoves(string[] fensAndMoves)
    {
      this.fensAndMoves = fensAndMoves.ToList();
    }

    /// <summary>
    /// Builds and verifies the full file name of a specified base file
    /// which is either a PGN or EPD.
    /// </summary>
    /// <param name="baseName"></param>
    /// <returns></returns>
    string GetFullFilename(string baseName)
    {
      // Accept the base name exactly as it is if it already exists
      if (System.IO.File.Exists(baseName))
        return baseName;

      string fullName;
      if (baseName.ToUpper().EndsWith("EPD"))
        fullName = Path.Combine(CeresUserSettingsManager.Settings.DirEPD, baseName);
      else if (baseName.ToUpper().EndsWith("PGN"))
        fullName = Path.Combine(CeresUserSettingsManager.Settings.DirPGN, baseName);
      else
        fullName = baseName;

      if (!System.IO.File.Exists(fullName))
        throw new Exception($"The source file {fullName} does not exist.");
      else
        return fullName;
    }


    /// <summary>
    /// Sets the opening source from a specified file name,
    /// which must be either an EPD or PGN file.
    /// </summary>
    /// <param name="fileName"></param>
    void LoadOpenings(string fileName)
    {
      if (fileName.ToUpper().EndsWith(".PGN"))
        LoadOpeningsPGN(fileName);
      else if (fileName.ToUpper().EndsWith(".EPD"))
        epds = EPDEntry.EPDEntriesInEPDFile(GetFullFilename(fileName));
      else
        throw new Exception($"PositionsWithHistory expected a file name with extension PGN or EPD ({fileName}).");
    }


    /// <summary>
    /// Sets the opening source from a specified PGN file.
    /// </summary>
    /// <param name="pgnFileName"></param>
    private void LoadOpeningsPGN(string pgnFileName)
    {
      openings = new List<PGNGame>();
      
      pgnFileName = GetFullFilename(pgnFileName);

      PgnStreamReader pgnReader = new PgnStreamReader();
      foreach (GameInfo game in pgnReader.Read(pgnFileName))
      {
        openings.Add(new PGNGame(game));
      }
    }

    /// <summary>
    /// Returns enumerator to iterate over the constituent PositionWithHistory.
    /// </summary>
    /// <returns></returns>
    public IEnumerator<PositionWithHistory> GetEnumerator()
    {
      return MoveSequences.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
      return ((IEnumerable)MoveSequences).GetEnumerator();
    }
  }

}
