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

using System.Text;

namespace Ceres.Chess.Textual.PgnFileTools
{
  public class Move
  {
    public int? Annotation { get; set; }
    public CastleType CastleType { get; set; }
    public string Comment { get; set; }
    public File DestinationFile { get; set; }
    public Row DestinationRow { get; set; }
    public string ErrorMessage { get; set; }
    public bool HasError { get; set; }
    public bool IsCapture { get; set; }
    public bool IsCastle { get; set; }
    public bool IsCheck { get; set; }
    public bool IsDoubleCheck { get; set; }
    public bool IsEnPassantCapture { get; set; }
    public bool IsMate { get; set; }
    public bool IsPromotion { get; set; }
    public int Number { get; set; }
    public PieceType PieceType { get; set; }
    public PieceType PromotionPiece { get; set; }
    public File SourceFile { get; set; }
    public Row SourceRow { get; set; }
    public string Variation { get; set; }

    public string ToAlgebraicString()
    {
      var result = new StringBuilder();
      if (PieceType != PieceType.Pawn)
      {
        if (IsCastle)
        {
          result.Append(CastleType.Symbol);
        }
        else
        {
          result.Append(PieceType.Symbol);
        }
      }
      if (SourceFile != null)
      {
        result.Append(SourceFile.Symbol);
      }
      if (SourceRow != null)
      {
        result.Append(SourceRow.Symbol);
      }
      if (IsCapture)
      {
        result.Append('x');
      }
      if (!IsCastle)
      {
        result.Append(DestinationFile.Symbol);
        result.Append(DestinationRow.Symbol);
      }
      if (IsPromotion)
      {
        result.Append('=');
        result.Append(PromotionPiece.Symbol);
      }
      else if (IsEnPassantCapture)
      {
        result.Append("ep");
      }

      if (IsCheck)
      {
        result.Append('+');
        if (IsDoubleCheck)
        {
          result.Append('+');
        }
      }
      else if (IsMate)
      {
        result.Append('#');
      }
      return result.ToString();
    }
  }
}
