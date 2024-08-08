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
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

using Ceres.Base.Math;
using Ceres.Base.DataTypes;
using Ceres.Base.DataType;

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.Chess.LC0.Boards;

#endregion

namespace Ceres.Chess.NNEvaluators.Ceres.TPG
{
  /// <summary>
  /// Array of policy probabilities.
  /// 
  /// Values are padded so last element is repeated as many times as needed.
  /// </summary>
  [InlineArray(TPGRecord.MAX_MOVES)]
  public struct PolicyValues92
  {
    private Half PolicyValue;
  }

  /// <summary>
  /// Array of indices of moves.
  /// Values are padded so last element is repeated as many times as needed.
  /// </summary>
  [InlineArray(TPGRecord.MAX_MOVES)]
  public struct PolicyIndices92
  {
    private short PolicyIndex;
  }


  /// <summary>
  /// Array of 64 TPGSquareRecord.
  /// </summary>
  [InlineArray(64)]
  public struct TPGSquareRecord64
  {
    private TPGSquareRecord Square;
  }


  /// <summary>
  /// Binary structure for one neural network training sample
  /// comprising a position, associated legal moves, and associated training targets.
  /// </summary>
  [Serializable]
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public unsafe struct TPGRecord
  {
    /// <summary>
    /// Currently hardcoded value for the per-square dimension of the prior state information, 
    /// if the network has a state output. Linked to TPGRecord.SIZE_STATE_PER_SQUARE.NNEvaluator
    /// </summary>
    public const int SIZE_STATE_PER_SQUARE = 4;

    /// <summary>
    /// If the feature that encodes number of ply since last move on each square is enabled.
    /// </summary>
    public const bool EMIT_PLY_SINCE_LAST_MOVE_PER_SQUARE = false;


    #region Data structure sizing

    /// <summary>
    /// Neural network input consists of description of all the 64 squares on chessboard.
    /// </summary>
    public const int NUM_SQUARES = 64;

    /// <summary>
    /// Maximum number of policy moves in a position that can be stored.
    /// NOTE: this value is hardcoded several places:
    ///   - ONNXRuntimeExecutor
    ///   - tpg_dataset.py in the Pytorch training code
    /// </summary>
    public const int MAX_MOVES = 92;

    // *** WARNING **** value hardcoded in ONNXRuntimeExecutor.TPG_BYTES_PER_SQUARE_RECORD currently, fix 
    public static int BYTES_PER_SQUARE_RECORD => Marshal.SizeOf<TPGSquareRecord>();

    public static int TOTAL_BYTES => Marshal.SizeOf<TPGRecord>();

    #endregion


    #region Data structure fields

    /// <summary>
    /// Win/draw/loss game result (without deblundering) as a 3-element array.
    /// TODO: this could be stored more simply as a single value (e.g. 0 = loss, 1 = draw, 2 = win).
    /// </summary>
    public fixed float WDLResultNonDeblundered[3];

    /// <summary>
    /// Win/draw/loss game result (with deblundering) as a 3-element array.
    /// </summary>
    public fixed float WDLResultDeblundered[3];

    /// <summary>
    /// Win/draw/loss training target as a 3-element array (derived from the root node of search Q value).
    /// </summary>
    public fixed float WDLQ[3];

    /// <summary>
    /// Estimated suboptimality of Q (in range 0 to 2) from the training data
    /// of this position relative to the estimated Q of the position which
    /// would have arisen if the best possible move had been played.
    /// </summary>
    public float PlayedMoveQSuboptimality;

    /// <summary>
    /// Training input if white was to move.
    /// This is just informational, since the training positions are 
    /// always from the perspective of the side to move.
    /// </summary>
    public byte IsWhiteToMove;
    public byte Unused1;
    public byte Unused2;
    public byte Unused3;
    public fixed byte UnusedArray[52];

    /// <summary>
    /// Kullback-Leibler divergence between policy head and search visits in nats.
    /// </summary>
    public float KLDPolicy;

    /// <summary>
    /// Moves (actually half-moves) left until game end training target.
    /// </summary>
    public float MLH;
    
    /// <summary>
    /// Difference between Q of best move and V as a training target proxying for uncertainty.
    /// We record this rather than (say) the absolute difference for uncertainty
    /// to maintain maximum flexibility since downstream consumers can apply additional transformations.
    /// </summary>
    public float DeltaQVersusV;

    /// <summary>
    /// Target forward Q deviation (lower bound).
    /// </summary>
    public Half QDeviationLower;

    /// <summary>
    /// Target forward Q deviation (upper bound).
    /// </summary>
    public Half QDeviationUpper;

    /// <summary>
    /// Neural net index (0...1857) of the move played from prior move in game (or -1 if none).
    /// </summary>
    public short PolicyIndexInParent;

    /// <summary>
    /// Policy training target with array of move indices having nozero probabilities.
    /// </summary>
    public PolicyIndices92 PolicyIndices;

    /// <summary>
    /// Policy training target with array of move probabilities corresponding to PolicyIndices.
    /// </summary>
    public PolicyValues92 PolicyValues;

    #endregion


    public readonly float SearchQ => WDLQ[0] - WDLQ[2];
    public readonly float ResultQDeblundered => WDLResultDeblundered[0] - WDLResultDeblundered[2];
    public readonly float ResultQNonDeblundered => WDLResultNonDeblundered[0] - WDLResultNonDeblundered[2];

    public readonly float[] WDLResultDeblunderedArray => new float[] { WDLResultDeblundered[0], WDLResultDeblundered[1], WDLResultDeblundered[2] };
    public readonly float[] WDLResultNonDeblunderedArray => new float[] { WDLResultNonDeblundered[0], WDLResultNonDeblundered[1], WDLResultNonDeblundered[2] };
    public readonly float[] WDLQResultArray => new float[] { WDLQ[0], WDLQ[1], WDLQ[2] };

    public readonly EncodedMove MoveAtIndex(int i) => EncodedMove.FromNeuralNetIndex(i);

    /// <summary>
    /// Array of 64 squares with piece information, history, etc.
    /// </summary>
    public TPGSquareRecord64 Squares;

    /// <summary>
    /// Copies all data relating TPGSquareRecord into a byte array
    /// starting at a specified offset in the array.
    /// </summary>
    public void CopySquares(byte[] array, int offset)
    {
      // Get a view of the data for whole record as span of bytes
      ReadOnlySpan<byte> bufferAsBytes = MemoryMarshal.Cast<TPGSquareRecord, byte>(Squares);
      bufferAsBytes.CopyTo(array.AsSpan().Slice(offset, NUM_SQUARES * BYTES_PER_SQUARE_RECORD));
    }


    /// <summary>
    /// Computes a hash value over all of the Squares interpreted as bytes.
    /// </summary>
    public long SquaresByteHash
    {
      get
      {
        long hash = 0;
        ReadOnlySpan<byte> bufferAsBytes = MemoryMarshal.Cast<TPGSquareRecord, byte>(Squares);
        for (int i = 0; i < NUM_SQUARES; i++)
        {
          for (int j = 0; j < BYTES_PER_SQUARE_RECORD; j++)
          {
            // update hash
            hash = MathUtils.HashCombineStable(hash, bufferAsBytes[i * BYTES_PER_SQUARE_RECORD + j]);
          }
        }

        return hash;
      }
    }


    /// <summary>
    /// Returns the number of squares for which pieces are the same
    /// between PieceTypeHistory0 and the other piece type histories.
    /// </summary>
    /// <returns></returns> 
    private readonly int[] GetPieceTypeHistoryEqualCounts()
    {
      int[] equalHistory = new int[TPGSquareRecord.NUM_HISTORY_POS];
      for (int i = 0; i < NUM_SQUARES; i++)
      {
        for (int h = 1; h < TPGSquareRecord.NUM_HISTORY_POS; h++)
        {
          if (ByteScaled.SpansEqual(Squares[i].PieceTypeHistory(h), Squares[i].PieceTypeHistory(0)))
          {
            equalHistory[h]++;
          }
        }
      }

      return equalHistory;
    }


    /// <summary>
    /// Dumps a string representation of the TPGRecord to the Console.
    /// </summary>
    public readonly void Dump(bool withHistory = true)
    {
      int[] equalHistoryCounts = GetPieceTypeHistoryEqualCounts();

      Console.WriteLine("\r\nPOSITIONS (current and history), to move " + (IsWhiteToMove > 0 ? "White" : "Black"));
      Console.WriteLine("  " + FinalPosition.FEN);
      if (withHistory)
      {
        for (int i = 1; i < TPGSquareRecord.NUM_HISTORY_POS; i++)
        {
          Console.WriteLine("  " + HistoryPosition(i).FEN + " " + equalHistoryCounts[i]);
        }
      }

      Console.WriteLine($"WDL            {WDLResultDeblunderedArray[0]:6,2} {WDLResultDeblunderedArray[1]:6,2} {WDLResultDeblunderedArray[2]:6,2}");
      Console.WriteLine($"WDLQ           {WDLQResultArray[0]:6,2} {WDLQResultArray[1]:6,2} {WDLQResultArray[2]:6,2}");
      Console.WriteLine($"MLH / UNC      {MLH:6,2}  {DeltaQVersusV:6,2}");

      for (int i = 0; i < NUM_SQUARES; i++)
      {
        Console.WriteLine($"Square {i:N2}  {Squares[i]}");
      }

      for (int i = 0; i < MAX_MOVES; i++)
      {
        if (i > 1 && PolicyIndices[i] == PolicyIndices[i - 1])
        {
          // Hit beginning of padding (repeated last move).
          break;
        }
        Console.WriteLine($"Policy[{i}]  {100 * Math.Round((float)PolicyValues[i], 5):N3}%  {MoveAtIndex(PolicyIndices[i])}");
      }
    }

#if NOT
    This is functional but has a dependency not currently satisfied (GraphvizUtils).

    /// <summary>
    /// Dumps information about TPGRecord (including history and moves between boards)
    /// then launches browser with a set of chessboards showing the position and history.
    /// </summary>
    /// <param name="tpgRecord"></param>
    public readonly void DumpPositionWithHistoryInBrowser()
    {
      Dump();
      Console.WriteLine();

      Console.WriteLine("Transitional moves between boards");
      for (int h = 0; h < 7; h++)
      {
        Console.WriteLine($"  {h} {MoveBetweenHistoryPositions(h + 1, h)}");
      }

      string fn = GraphvizUtils.WritePositionsToSVGFile(true,
                                                        (HistoryPosition(0), "pos0"),
                                                        (HistoryPosition(1), "pos1"),
                                                        (HistoryPosition(2), "pos2"),
                                                        (HistoryPosition(3), "pos3"),
                                                        (HistoryPosition(4), "pos4"),
                                                        (HistoryPosition(5), "pos5"),
                                                        (HistoryPosition(6), "pos6"),
                                                        (HistoryPosition(7), "pos7")
                                                        );
      StringUtils.LaunchBrowserWithURL(fn);
    }
#endif

    /// <summary>
    /// Returns the current Position corresponding to to this TPGRecord.
    /// </summary>
    public readonly Position FinalPosition => PositionForSquares(Squares, 0, IsWhiteToMove > 0, true);


    /// <summary>
    /// Getters for history positions (prior to current position).
    /// N.B. History positions extracted here may not be complete (do not contain correct EP, castling rights etc.).
    /// </summary>
    public readonly Position HistoryPosition(int historyIndex, bool tryComputeEnPassant = true) => PositionForSquares(Squares, historyIndex, IsWhiteToMove > 0, tryComputeEnPassant);


    /// <summary>
    /// Returns full history of positions for this TPGRecord.
    /// </summary>
    /// <param name="maxHistoryPositions"></param>
    /// <returns></returns>
    public readonly PositionWithHistory ToPositionWithHistory(int maxHistoryPositions = EncodedPositionBoards.NUM_MOVES_HISTORY)
    {
      List<Position> tpgPositions = new(maxHistoryPositions);
      Position lastPosition = default;
      for (int i = 0; i < maxHistoryPositions; i++)
      {
        Position thisPos = HistoryPosition(i);

        // TPG may already have the history filled in.
        // Detect this and omit the fill-in positions.
        bool isHistoryFillIn = thisPos.PiecesEqual(in lastPosition);
        if (!isHistoryFillIn)
        {
          tpgPositions.Add(thisPos);
        }

        lastPosition = thisPos;
      }

      tpgPositions.Reverse();
      return new PositionWithHistory(tpgPositions, true);
    }


    /// <summary>
    /// Returns the move played between two history boards.
    /// </summary>
    /// <param name="priorIndex"></param>
    /// <param name="curIndex"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public readonly MGMove MoveBetweenHistoryPositions(int priorIndex, int curIndex)
    {
      if (priorIndex < curIndex)
      {
        throw new Exception("priorIndex must be greater than curIndex");
      }

      Position priorPos = HistoryPosition(priorIndex);
      Position curPos = HistoryPosition(curIndex);

      MGMoveList moves = new MGMoveList();
      MGMoveGen.GenerateMoves(priorPos.ToMGPosition, moves);

      for (int i = 0; i < moves.NumMovesUsed; i++)
      {
        Position newPos = priorPos.AfterMove(MGMoveConverter.ToMove(moves.MovesArray[i]));
        if (newPos.PiecesEqual(curPos))
        {
          return moves.MovesArray[i];
        }
      }
      return default;
    }


    /// <summary>
    /// Returns the Position corresponding to the given squares and piece history index.
    /// 
    /// N.B. Only the position at index 0 is guaranteed to be the exactly correct position,
    ///      the others may be incomplete (with missing/incorrect EP, castling rights etc.).
    ///      
    /// TODO: Try to fix this, for example EP can be derived for all except earliest history position.
    /// </summary>
    /// <param name="squares"></param>
    /// <param name="pieceHistoryIndex"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public readonly Position PositionForSquares(ReadOnlySpan<TPGSquareRecord> squares, int pieceHistoryIndex, bool tpgRecordIsWhiteToMove, bool tryComputeEnPassant)
    {
      Debug.Assert(squares.Length == NUM_SQUARES);

      bool weAreWhite = tpgRecordIsWhiteToMove == (pieceHistoryIndex % 2 == 0);

      Span<PieceOnSquare> piecesOnSquares = stackalloc PieceOnSquare[32];

      int validSquareRecordIndex = default;
      bool needsReversal = !tpgRecordIsWhiteToMove;
      PositionMiscInfo.EnPassantFileIndexEnum enPassant = PositionMiscInfo.EnPassantFileIndexEnum.FileNone;
      int numAdded = 0;
      for (int i = 0; i < 64; i++)
      {
        ReadOnlySpan<ByteScaled> pieceTypeHistory = squares[i].PieceTypeHistory(pieceHistoryIndex);
        (PieceType pieceType, bool isOurPiece) pt = TPGSquareRecord.GetPieceInfo(pieceTypeHistory);
        bool isWhitePiece = pt.isOurPiece == tpgRecordIsWhiteToMove;
        if (pt.pieceType != PieceType.None)
        {
          Piece piece = isWhitePiece ? new Piece(SideType.White, pt.pieceType)
                                      : new Piece(SideType.Black, pt.pieceType);
          Square square = needsReversal ? squares[i].GetSquare().Reversed : squares[i].GetSquare();
          piecesOnSquares[numAdded++] = new PieceOnSquare(square, piece);

          // In the case of the last position, the en passant information is available.
          if (pieceHistoryIndex == 0 && squares[i].IsEnPassant.Value == 1)
          {
            Debug.Assert(piece.Type == PieceType.Pawn);
            Debug.Assert(square.Rank == (weAreWhite ? 4 : 3));
            enPassant = (PositionMiscInfo.EnPassantFileIndexEnum)square.File;
          }
          validSquareRecordIndex = i; // any index occupied will suffice for getting misc info
        }
      }

      // For first position (0), en passant is explicitly recorded and captured in above loop.
      // For last position (7) there is no way to check for en passant opportunity since the prior board is not available.
      // However for other positions, we can here infer the opportunity from the placement of pieces on the two boards.
      if (tryComputeEnPassant && pieceHistoryIndex > 0 && pieceHistoryIndex < 7)
      {
        enPassant = TPGRecordUtils.EnPassantOpportunityBetweenTPGRecords(this, pieceHistoryIndex + 1, pieceHistoryIndex);
      }

      TPGSquareRecord squareRecord = squares[validSquareRecordIndex];

      bool isWhite = pieceHistoryIndex % 2 == 0 == tpgRecordIsWhiteToMove;

      bool whiteCanOO, whiteCanOOO, blackCanOO, blackCanOOO;
      if (pieceHistoryIndex == 0)
      {
        whiteCanOO = isWhite ? squareRecord.CanOO.Value == 1 : squareRecord.OpponentCanOO.Value == 1;
        whiteCanOOO = isWhite ? squareRecord.CanOOO.Value == 1 : squareRecord.OpponentCanOOO.Value == 1;
        blackCanOO = isWhite ? squareRecord.OpponentCanOO.Value == 1 : squareRecord.CanOO.Value == 1;
        blackCanOOO = isWhite ? squareRecord.OpponentCanOOO.Value == 1 : squareRecord.CanOOO.Value == 1;
      }
      else
      {
        // Castling information not directly recorded. 
        // Assume it is possible to allow move generation from this position to find castle if it was in fact possible.
        // TODO: someday make this smarter to look at rook/king position to determine if possible?
        whiteCanOO = whiteCanOOO = blackCanOO = blackCanOOO = true;
      }

      int move50Count = pieceHistoryIndex > 0 ? 0 : TPGRecordEncoding.Move50CountDecoded(squareRecord.Move50Count.Value);

      // Note that castling status and move 50 rule values are not available for positions other than current (index 0).
      // For index > 0, we substitute a guess based on the settings for the current position.
      PositionMiscInfo miscInfo = new PositionMiscInfo(whiteCanOO, whiteCanOOO, blackCanOO, blackCanOOO,
                                                       isWhite ? SideType.White : SideType.Black,
                                                       move50Count,
                                                       (int)squareRecord.HistoryRepetitionCounts[pieceHistoryIndex].Value,
                                                       2, // use move plies 2 (i.e. 1 full move) since true value unknown
                                                       enPassant);
      Position p = new Position(piecesOnSquares.Slice(0, numAdded), miscInfo);

      return p;
    }


    /// <summary>
    /// Static version of PositionForSquares (does not support tryCompteEnPassant).
    /// </summary>
    /// <param name="squares"></param>
    /// <param name="pieceHistoryIndex"></param>
    /// <param name="tpgRecordIsWhiteToMove"></param>
    /// <returns></returns>
    public static Position PositionForSquares(ReadOnlySpan<TPGSquareRecord> squares, int pieceHistoryIndex, bool tpgRecordIsWhiteToMove)
    {
      // Since last argument (tryComputeEnPassant) is false, this can be called as if static (default object).
      return default(TPGRecord).PositionForSquares(squares, pieceHistoryIndex, tpgRecordIsWhiteToMove, false);
    }



    /// <summary>
    /// Returns if a policy move with the given index exists in the policy.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public readonly bool PolicyWithIndexExists(int index)
    {
      for (int i = 0; i < MAX_MOVES; i++)
      {
        if (i > 1 && PolicyIndices[i] == PolicyIndices[i - 1])
        {
          // Hit beginning of padding (repeated last move).
          break;
        }

        if (PolicyIndices[i] == index)
        {
          return true;
        } 
      }

      return false;
    } 


    /// <summary>
    /// Extracts the policies into a CompressedPolicyVector.
    /// </summary>
    /// <param name="policy"></param>
    public unsafe readonly CompressedPolicyVector PolicyVector
    {
      get
      {
        // Create span of indices converted to type int and adjusted to be zero-based.
        Span<int> indices = stackalloc int[MAX_MOVES];
        Span<float> probs = stackalloc float[MAX_MOVES]; // TODO: convert to FP16?

        int i = 0;
        int lastPolicyIndex = -1;
        while (i < MAX_MOVES)
        {
          int thisIndex = PolicyIndices[i];
          if (thisIndex == lastPolicyIndex)
          {
            // Starting to repeat, have reached end.
            break;
          }

          indices[i] = thisIndex;
          probs[i] = (FP16)(float)PolicyValues[i];

          lastPolicyIndex = thisIndex;
          i++;
        }

        // Convert to policy vector.
        CompressedPolicyVector policy = default;
        CompressedPolicyVector.Initialize(ref policy, indices.Slice(0, i), probs.Slice(0, i), false);
        return policy;
      }
    }


    /// <summary>
    /// Validates the expected fixed sizing of the TPGRecord data structure.
    /// </summary>
    public static void Validate()
    {
      Debug.Assert(Marshal.SizeOf<TPGSquareRecord>() == BYTES_PER_SQUARE_RECORD);
      //      Console.WriteLine(Marshal.SizeOf<TPGRecord>());
      //      Console.WriteLine(TOTAL_BYTES);
      Debug.Assert(TOTAL_BYTES == Marshal.SizeOf<TPGRecord>());
    }
  }

}
