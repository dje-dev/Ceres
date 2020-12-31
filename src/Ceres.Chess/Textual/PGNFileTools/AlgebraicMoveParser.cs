#region License notice

/*
Adapted from the PgnFileTools project by Clinton Sheppard 
at https://github.com/handcraftsman/PgnFileTools
licensed under Apache License, Version 2.
*/

#endregion

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
using System.Text.RegularExpressions;

#endregion

namespace Ceres.Chess.Textual.PgnFileTools
{
  public class AlgebraicMoveParser
  {
    private const string Capture = "capture";
    private const string Castle = "castle";
    private const string Check = "check";
    private const string DestinationFile = "destinationfile";
    private const string Destinationrow = "destinationrow";
    private const string DoubleCheck = "doublecheck";
    private const string EnPassant = "enpassant";
    private const string Mate = "mate";
    private const string Piece = "piece";
    private const string Promotionpiece = "promotionpiece";
    private const string SourceFile = "sourcefile";
    private const string SourceRow = "sourcerow";
    private static readonly Regex _regex;

    static AlgebraicMoveParser()
    {
      _regex = new Regex(
          "^(((?<" + Piece + ">[NBRQK])?(?<" + SourceFile + ">[a-h])?(?<" + SourceRow + ">[1-8])?(?<" + Capture + ">x)?(?<" + DestinationFile + ">[a-h])(?<" + Destinationrow + ">[1-8]))|" +
          "(((?<" + Piece + ">[NBRQK])?(?<" + SourceFile + ">[a-h])?(?<" + SourceRow + ">[1-8])?(?<" + Capture + ">x)(?<" + DestinationFile + ">[a-h])(?<" + Destinationrow + ">[36]))(?<" + EnPassant + ">ep))|" +
          "((?<" + SourceFile + ">[a-h])?(?<" + Capture + ">x)?(?<" + DestinationFile + ">[a-h])(?<" + Destinationrow + ">[2-7]))|" +
          "((?<" + SourceFile + ">[a-h])(?<" + Capture + ">x)(?<" + DestinationFile + ">[a-h])(?<" + Destinationrow + ">[36])(?<" + EnPassant + ">ep))|" +
          "((?<" + SourceFile + ">[a-h])?(?<" + Capture + ">x)?(?<" + DestinationFile + ">[a-h])(?<" + Destinationrow + @">[18])(?:\=?(?<" + Promotionpiece + ">[NBRQK])))|" +
          "(?<" + Castle + ">O(-?O){1,2}))" +
          "((?<" + Check + @">\+)(?<" + DoubleCheck + @">\+)?|(?<" + Mate + ">#))?$",
          RegexOptions.Compiled | RegexOptions.ExplicitCapture | RegexOptions.CultureInvariant);
    }

    public Move Parse(string algebraicMove)
    {
      var match = _regex.Match(algebraicMove);

      var move = new Move();
      if (match.Length == 0)
      {
        move.HasError = true;
        move.ErrorMessage = "Failed to match expected structure";
        return move;
      }

      move.CastleType = CastleType.GetFor(match.Groups[Castle].Value);
      move.IsCastle = move.CastleType != null;
      if (!move.IsCastle)
      {
        move.PieceType = PieceType.GetFor(match.Groups[Piece].Value);
        move.SourceFile = match.Groups[SourceFile].Value.Length == 0 ? null : File.GetFor(match.Groups[SourceFile].Value[0]);
        move.SourceRow = Row.GetFor(match.Groups[SourceRow].Value);
        move.IsCapture = !match.Groups[Capture].Value.IsNullOrEmptyExtensions();
        move.DestinationFile = match.Groups[DestinationFile].Value.Length == 0 ? null : File.GetFor(match.Groups[DestinationFile].Value[0]);
        move.DestinationRow = Row.GetFor(match.Groups[Destinationrow].Value);
        if (move.PieceType == PieceType.Pawn)
        {
          move.PromotionPiece = match.Groups[Promotionpiece].Value.Length == 0 ? null : PieceType.GetFor(match.Groups[Promotionpiece].Value[0]);
          move.IsPromotion = move.PromotionPiece != null;
        }
        if (move.IsCapture)
        {
          move.IsEnPassantCapture = !match.Groups[EnPassant].Value.IsNullOrEmptyExtensions();
        }
        if (!move.PieceType.IsLegal(new Position
        {
          File = move.SourceFile,
          Row = move.SourceRow
        }, move.IsCapture, new Position
        {
          File = move.DestinationFile,
          Row = move.DestinationRow
        }))
        {
          move.HasError = true;
          move.ErrorMessage = "The source and destination are illegal for this piece type";
        }
      }

      move.IsCheck = !match.Groups[Check].Value.IsNullOrEmptyExtensions();
      if (move.IsCheck)
      {
        move.IsDoubleCheck = !match.Groups[DoubleCheck].Value.IsNullOrEmptyExtensions();
      }
      else
      {
        move.IsMate = !match.Groups[Mate].Value.IsNullOrEmptyExtensions();
      }

      return move;
    }
  }

  public static class ExtensionsX
    {
    public static bool IsNullOrEmptyExtensions(this string input, bool trim = false)
    {
      if (String.IsNullOrEmpty(input))
      {
        return true;
      }
      return trim && String.IsNullOrEmpty(input.Trim());
    }

  }
}
