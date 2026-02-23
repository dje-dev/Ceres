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

using Ceres.Base.DataType;

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

#endregion

namespace Ceres.Chess.NNEvaluators.Ceres.TPG
{
  /// <summary>
  /// Static helper methods which validate integrity of TPGRecords.
  /// </summary>
  public class TPGRecordValidation
  {
    /// <summary>
    /// Validates basic aspects of TPG integrity, such as all squares having valid piece types.
    /// </summary>
    /// <param name="tpgRecord"></param>
    /// <exception cref="Exception"></exception>
    /// <exception cref="NotImplementedException"></exception>
    static void ValidateBasic(in TPGRecord tpgRecord)
    {
      for (int historyPlane = 0; historyPlane < 8; historyPlane++)
      {
        bool found = false;
        foreach (ByteScaled b in tpgRecord.Squares[historyPlane].PieceTypeHistory(historyPlane))
        {
          if (b.Value != 0)
          {
            found = true;
            break;
          }
        }
        if (!found)
        {
          throw new Exception($"History plane {historyPlane} had no piece/blank set");
        }
      }

      for (int i = 0; i < 64; i++)
      {
        ByteScaled.ValidateZeroOrOne(tpgRecord.Squares[i].RankEncoding);
        ByteScaled.ValidateZeroOrOne(tpgRecord.Squares[i].FileEncoding);
        if (!(tpgRecord.Squares[i].IsEnPassant.IsZeroOrOne
             && tpgRecord.Squares[i].CanOO.IsZeroOrOne
             && tpgRecord.Squares[i].CanOOO.IsZeroOrOne
             && tpgRecord.Squares[i].OpponentCanOO.IsZeroOrOne
             && tpgRecord.Squares[i].OpponentCanOOO.IsZeroOrOne
             )
            )
        {
          throw new NotImplementedException("Expected binary value not binary");
        }
      }
    }


    /// <summary>
    /// Verifies that the converted tpgRecord is valid
    /// (e.g. positions representations are equivalent).
    /// </summary>
    /// <param name="encodedPosition"></param>
    /// <param name="tpgRecord"></param>
    /// <param name="validatePolicy"></param>
    public static unsafe void Validate(in EncodedPositionWithHistory encodedPosition,
                                       in TPGRecord tpgRecord,
                                       bool validatePolicy = true)
    {
      Validate(in tpgRecord, validatePolicy);

      Position posTraining = encodedPosition.FinalPosition;
      Position posTPG = tpgRecord.FinalPosition;
      if (!posTraining.PiecesEqual(in posTPG))
      {
        Console.WriteLine("Internal error! TPG conversion does not match. " + posTraining.FEN + " " + posTPG.FEN);
      }

      if (validatePolicy)
      {
        ValidateMoves(in posTraining, in tpgRecord, "training (source)");
      }

    }


    /// <summary>
    /// Verifies that the converted tpgRecord is valid
    /// (e.g. positions representations are equivalent).
    /// </summary>
    /// <param name="encodedPosition"></param>
    /// <param name="tpgRecord"></param>
    /// <param name="validatePolicy"></param>
    public static unsafe void Validate(in TPGRecord tpgRecord, bool validatePolicy = true)
    {
      ValidateBasic(in tpgRecord);
      ValidateHistoryReachability(in tpgRecord);

      if (validatePolicy)
      {
        ValidateMoves(tpgRecord.FinalPosition, in tpgRecord, "TPG (target)");
      }
    }


    /// <summary>
    /// Verifies that the converted tpgRecord is valid
    /// (e.g. checking that all training position moves are valid from the converted position).
    /// </summary>
    /// <param name="trainingPos"></param>
    /// <param name="tpgRecord"></param>
    /// <exception cref="Exception"></exception>
    public static unsafe void ValidateSquares(in EncodedPositionWithHistory trainingPos, ref TPGRecord tpgRecord)
    {
      Position posTraining = trainingPos.FinalPosition;
      Position posTPG = tpgRecord.FinalPosition;// trainingPos.MiscInfo.WhiteToMove ? tpgRecord.Position : tpgRecord.Position.Reversed;
      if (!posTraining.PiecesEqual(in posTPG))
      {
        throw new Exception("Internal error! TPG conversion does not match. " + posTraining.FEN + " " + posTPG.FEN);
      }
    }


    static int errorCount = 0;

    /// <summary>
    /// Verifies that all the moves with nonzero probability in the training position
    /// are moves which are legal from the specified position.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="tpgRecord"></param>
    /// <param name="desc"></param>
    static unsafe void ValidateMoves(in Position pos, in TPGRecord tpgRecord, string desc)
    {
      MGMoveList legalMoves = new MGMoveList();
      MGMoveGen.GenerateMoves(pos.ToMGPosition, legalMoves);
      for (int i = 0; i < TPGRecord.MAX_MOVES; i++)
      {
        EncodedMove em = EncodedMove.FromNeuralNetIndex(tpgRecord.PolicyIndices[i]);
        MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(em, pos.ToMGPosition);
        if (!legalMoves.MoveExists(mgMove))
        {
#if NOT
          if (errorCount++ < 3) // avoid opening too many browser windows
          {
            tpgRecord.DumpPositionWithHistoryInBrowser();
          }
          else
#endif
          tpgRecord.Dump();
          Console.WriteLine("Illegal move found in converted training data " + pos.FEN + " " + em + " " + mgMove + " " + desc);
        }
      }

    }

    /// <summary>
    /// Validates that sequence of history positions are valid,
    /// i.e. reachable from each other in on move.
    /// </summary>
    /// <param name="tpgRecord"></param>
    public static void ValidateHistoryReachability(in TPGRecord tpgRecord)
    {
      for (int i = 1; i < 8; i++)
      {
        ValidateHistoryReachability(in tpgRecord, i);
      }
    }


    /// <summary>
    /// Validates that a pair of history positions are reachable from each other in one move.
    /// </summary>
    /// <param name="tpgRecord"></param>
    /// <param name="priorIndex"></param>
    /// <exception cref="Exception"></exception>
    static unsafe void ValidateHistoryReachability(in TPGRecord tpgRecord, int priorIndex)
    {
      Position posCur = tpgRecord.HistoryPosition(priorIndex - 1);
      Position posPrior = tpgRecord.HistoryPosition(priorIndex);
      bool ok = posPrior.PieceCount == 0 || MGPositionReachability.IsProbablyReachable(posPrior.ToMGPosition, posCur.ToMGPosition);
      if (!ok)
      {
        bool priorLooksLikeFillIn = posCur.PiecesEqual(posPrior.Reversed);
        if (!priorLooksLikeFillIn)
        {
          // TODO: consider also verifying that once a fill-in appears it then appears for all subsequent positions.
          tpgRecord.Dump();
          throw new Exception("At history index " + priorIndex + " position does not look reachable from prior position " + posPrior.FEN + " --> " + posCur.FEN);
        }
      }
    }

  }

}
