using System;
using System.Collections.Generic;
using System.Linq;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.Positions;

namespace Ceres.MCGS.Visualization.PlyVisualization;

record PVCandidate(Move[] Moves, MGMove[] MGMoves, double LogScore, double PolicyScore, double PlyBinScore);

static class PVExtractor
{
  const double W_POLICY = 1.0;
  const double W_PLYBIN = 0.5;
  const double W_CAPTURE = 0.3;
  const double W_PUNIM = 0.15;
  const double EPS = 1e-8;
  const int DEFAULT_MAX_DEPTH = 5;
  const int DEFAULT_BEAM_WIDTH = 12;
  const int DEFAULT_TOP_K = 3;
  const float MIN_POLICY_PROB = 0.005f;

  // Beam candidate during search.
  struct BeamEntry
  {
    public MGMove[] Moves;
    public MGPosition Position;
    public double CumulativeScore;
    public double PolicyScore;
    public double PlyBinScore;
    public int Depth;
  }


  public static List<PVCandidate> ExtractPVs(NNEvaluatorResult rootEval, Position rootPosition,
                                              int maxDepth = DEFAULT_MAX_DEPTH,
                                              int beamWidth = DEFAULT_BEAM_WIDTH,
                                              int topK = DEFAULT_TOP_K)
  {
    if (rootEval.PlyBinMoveProbs == null)
    {
      return new List<PVCandidate>();
    }

    Half[] moveProbs = rootEval.PlyBinMoveProbs;
    Half[] captureProbs = rootEval.PlyBinCaptureProbs;
    Half[] punimSelf = rootEval.PunimSelfProbs;
    Half[] punimOpp = rootEval.PunimOpponentProbs;

    MGPosition rootMgPos = MGPosition.FromPosition(in rootPosition);

    // Collect top policy moves as seeds (one per PV).
    List<(MGMove move, float prob, double score, MGPosition pos)> seeds = new();
    foreach ((EncodedMove encodedMove, float prob) in rootEval.Policy.ProbabilitySummary(MIN_POLICY_PROB, Math.Max(beamWidth, topK * 4)))
    {
      MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(encodedMove, in rootMgPos);
      double moveScore = ScoreMove(moveProbs, captureProbs, punimSelf, punimOpp, mgMove, 1, prob);
      MGPosition newPos = MGPosition.MGPosAfterMove(rootMgPos, mgMove);
      seeds.Add((mgMove, prob, moveScore, newPos));
    }

    if (seeds.Count == 0)
    {
      return new List<PVCandidate>();
    }

    // Run a separate beam search per seed move to guarantee diverse PVs.
    List<PVCandidate> results = new(topK);
    MGMoveList moveList = new();
    int topKPerCandidate = 5;

    for (int seedIdx = 0; seedIdx < seeds.Count && results.Count < topK; seedIdx++)
    {
      (MGMove seedMove, float seedProb, double seedScore, MGPosition seedPos) = seeds[seedIdx];

      // Seed beam with just this one first move.
      List<BeamEntry> beam = new()
      {
        new BeamEntry
        {
          Moves = [seedMove],
          Position = seedPos,
          CumulativeScore = seedScore,
          PolicyScore = Math.Log(seedProb + EPS),
          PlyBinScore = seedScore - W_POLICY * Math.Log(seedProb + EPS),
          Depth = 1
        }
      };

      // Extend beam from depth 2 to maxDepth.
      for (int depth = 2; depth <= maxDepth; depth++)
      {
        List<BeamEntry> nextCandidates = new();

        foreach (BeamEntry candidate in beam)
        {
          MGMoveGen.GenerateMoves(in candidate.Position, moveList);
          if (moveList.NumMovesUsed == 0)
          {
            nextCandidates.Add(candidate);
            continue;
          }

          Span<(int idx, double score)> scored = moveList.NumMovesUsed <= 128
            ? stackalloc (int, double)[moveList.NumMovesUsed]
            : new (int, double)[moveList.NumMovesUsed];

          for (int m = 0; m < moveList.NumMovesUsed; m++)
          {
            MGMove mgMove = moveList.MovesArray[m];
            double moveScore = ScoreMove(moveProbs, captureProbs, punimSelf, punimOpp, mgMove, depth, 0);
            scored[m] = (m, moveScore);
          }

          int keepCount = Math.Min(topKPerCandidate, moveList.NumMovesUsed);
          PartialSort(scored, keepCount);

          for (int k = 0; k < keepCount; k++)
          {
            (int idx, double moveScore) = scored[k];
            MGMove mgMove = moveList.MovesArray[idx];
            MGPosition newPos = MGPosition.MGPosAfterMove(candidate.Position, mgMove);

            MGMove[] newMoves = new MGMove[candidate.Depth + 1];
            Array.Copy(candidate.Moves, newMoves, candidate.Depth);
            newMoves[candidate.Depth] = mgMove;

            nextCandidates.Add(new BeamEntry
            {
              Moves = newMoves,
              Position = newPos,
              CumulativeScore = candidate.CumulativeScore + moveScore,
              PolicyScore = candidate.PolicyScore,
              PlyBinScore = candidate.PlyBinScore + moveScore,
              Depth = depth
            });
          }
        }

        nextCandidates.Sort((a, b) => b.CumulativeScore.CompareTo(a.CumulativeScore));
        if (nextCandidates.Count > beamWidth)
        {
          nextCandidates.RemoveRange(beamWidth, nextCandidates.Count - beamWidth);
        }
        beam = nextCandidates;
      }

      // Take best PV from this seed's beam.
      if (beam.Count > 0)
      {
        beam.Sort((a, b) => b.CumulativeScore.CompareTo(a.CumulativeScore));
        BeamEntry best = beam[0];
        Move[] moves = new Move[best.Depth];
        for (int j = 0; j < best.Depth; j++)
        {
          moves[j] = MGMoveConverter.ToMove(best.Moves[j]);
        }
        MGMove[] mgMoves = new MGMove[best.Depth];
        Array.Copy(best.Moves, mgMoves, best.Depth);
        results.Add(new PVCandidate(moves, mgMoves, best.CumulativeScore, best.PolicyScore, best.PlyBinScore));
      }
    }

    return results;
  }


  public static List<PVCandidate>[] ExtractPVsBatch(NNEvaluatorResult[] rootEvals, Position[] rootPositions,
                                                     int maxDepth = DEFAULT_MAX_DEPTH,
                                                     int beamWidth = DEFAULT_BEAM_WIDTH,
                                                     int topK = DEFAULT_TOP_K)
  {
    List<PVCandidate>[] results = new List<PVCandidate>[rootEvals.Length];
    for (int i = 0; i < rootEvals.Length; i++)
    {
      results[i] = ExtractPVs(rootEvals[i], rootPositions[i], maxDepth, beamWidth, topK);
    }
    return results;
  }


  static int PlyToBin(int ply)
  {
    if (ply <= 2) return 0;
    if (ply <= 4) return 1;
    if (ply <= 10) return 2;
    if (ply <= 22) return 3;
    if (ply <= 40) return 4;
    if (ply <= 65) return 5;
    return 6;
  }


  static float SquareScore(Half[] plyBinProbs, int squareA1, int ply)
  {
    int bin = PlyToBin(ply);
    return (float)plyBinProbs[squareA1 * 8 + bin];
  }


  static double ScoreMove(Half[] moveProbs, Half[] captureProbs,
                           Half[] punimSelf, Half[] punimOpp,
                           MGMove move, int ply, float policyProb)
  {
    int fromA1 = move.FromSquare.SquareIndexStartA1;
    int toA1 = move.ToSquare.SquareIndexStartA1;

    // PlyBin move score: geometric mean of from and to square probabilities.
    float fromScore = SquareScore(moveProbs, fromA1, ply);
    float toScore = SquareScore(moveProbs, toA1, ply);
    double plyBinMoveScore;

    if (move.CastleShort || move.CastleLong)
    {
      // Castling: 4 squares (king from/to + rook from/to).
      (int rookFromA1, int rookToA1) = GetCastleRookSquares(move);
      float rookFromScore = SquareScore(moveProbs, rookFromA1, ply);
      float rookToScore = SquareScore(moveProbs, rookToA1, ply);
      double geomMean4 = Math.Pow((fromScore + EPS) * (toScore + EPS) *
                                   (rookFromScore + EPS) * (rookToScore + EPS), 0.25);
      plyBinMoveScore = W_PLYBIN * Math.Log(geomMean4 + EPS);
    }
    else
    {
      double geomMean2 = Math.Sqrt((fromScore + EPS) * (toScore + EPS));
      plyBinMoveScore = W_PLYBIN * Math.Log(geomMean2 + EPS);
    }

    // Policy score (depth 1 only).
    double policyScore = policyProb > 0 ? W_POLICY * Math.Log(policyProb + EPS) : 0;

    // Capture bonus from PlyBin capture head.
    double captureScore = 0;
    if (move.Capture)
    {
      int captureSquareA1 = toA1;
      if (move.EnPassantCapture)
      {
        // En passant: captured pawn is on the same file as to-square but on from-square's rank.
        int epFile = toA1 % 8;
        int epRank = fromA1 / 8;
        captureSquareA1 = epFile + 8 * epRank;
      }
      float capScore = SquareScore(captureProbs, captureSquareA1, ply);
      captureScore = W_CAPTURE * Math.Log(capScore + EPS);
    }

    // PUNIM score: irreversible moves (capture or pawn) should have high PUNIM probability
    // for current ply bin; reversible moves benefit from PUNIM predicting later.
    double punimScore = 0;
    if (punimSelf != null)
    {
      bool irreversible = move.Capture || IsPawnMove(move);
      int bin = PlyToBin(ply);
      if (irreversible)
      {
        // Cumulative probability up to this bin.
        double cumProb = 0;
        for (int b = 0; b <= bin; b++)
        {
          cumProb += (float)punimSelf[b];
        }
        punimScore = W_PUNIM * Math.Log(cumProb + EPS);
      }
      else
      {
        // Reversible: probability of irreversible move being later.
        double laterProb = 0;
        for (int b = bin + 1; b < 8; b++)
        {
          laterProb += (float)punimSelf[b];
        }
        punimScore = W_PUNIM * Math.Log(laterProb + EPS);
      }
    }

    return policyScore + plyBinMoveScore + captureScore + punimScore;
  }


  static bool IsPawnMove(MGMove move)
  {
    return move.Piece == MGPositionConstants.MCChessPositionPieceEnum.WhitePawn ||
           move.Piece == MGPositionConstants.MCChessPositionPieceEnum.BlackPawn;
  }


  static (int rookFromA1, int rookToA1) GetCastleRookSquares(MGMove move)
  {
    // Determine rook squares from king's from-square and castle direction.
    int kingFromA1 = move.FromSquare.SquareIndexStartA1;
    int rank = kingFromA1 / 8;

    if (move.CastleShort)
    {
      // Kingside: rook H-file -> F-file.
      return (7 + 8 * rank, 5 + 8 * rank);
    }
    else
    {
      // Queenside: rook A-file -> D-file.
      return (0 + 8 * rank, 3 + 8 * rank);
    }
  }


  public static string FormatPVasSAN(PVCandidate pv, Position startPos)
  {
    if (pv.MGMoves == null || pv.MGMoves.Length == 0)
    {
      return "";
    }

    List<string> parts = new();
    MGPosition mgPos = MGPosition.FromPosition(in startPos);
    MGMoveList disambigMoveList = new();
    int plyOffset = startPos.SideToMove == SideType.White ? 0 : 1;

    for (int i = 0; i < pv.MGMoves.Length; i++)
    {
      int ply = i + plyOffset;
      int moveNum = ply / 2 + 1;

      if (ply % 2 == 0)
      {
        parts.Add($"{moveNum}.");
      }
      else if (i == 0)
      {
        parts.Add($"{moveNum}...");
      }

      string san = FormatMGMoveAsSAN(pv.MGMoves[i], in mgPos, disambigMoveList);
      parts.Add(san);
      mgPos = MGPosition.MGPosAfterMove(mgPos, pv.MGMoves[i]);
    }

    return string.Join(" ", parts);
  }


  static string FormatMGMoveAsSAN(MGMove move, in MGPosition pos, MGMoveList moveList)
  {
    if (move.CastleShort) return "O-O";
    if (move.CastleLong) return "O-O-O";

    int pieceType = (byte)move.Piece & 7;
    bool isPawn = pieceType == MGPositionConstants.WPAWN;

    int fromA1 = move.FromSquare.SquareIndexStartA1;
    int toA1 = move.ToSquare.SquareIndexStartA1;
    int fromFile = fromA1 % 8;
    int toFile = toA1 % 8;
    int toRank = toA1 / 8;

    // Detect capture from position: check if target square is occupied.
    ulong targetBit = 1UL << move.ToSquareIndex;
    bool isCapture = ((pos.A | pos.B | pos.C | pos.D) & targetBit) != 0 || move.EnPassantCapture;

    string pieceStr = pieceType switch
    {
      MGPositionConstants.WKNIGHT => "N",
      MGPositionConstants.WBISHOP => "B",
      MGPositionConstants.WROOK => "R",
      MGPositionConstants.WQUEEN => "Q",
      MGPositionConstants.WKING => "K",
      _ => ""
    };

    // Disambiguation for non-pawn, non-king pieces.
    string disambig = "";
    if (!isPawn && pieceStr.Length > 0 && pieceType != MGPositionConstants.WKING)
    {
      MGMoveGen.GenerateMoves(in pos, moveList);
      bool needFile = false, needRank = false;
      for (int m = 0; m < moveList.NumMovesUsed; m++)
      {
        MGMove other = moveList.MovesArray[m];
        if (other.FromSquare.SquareIndexStartA1 == fromA1) continue;
        if (other.ToSquare.SquareIndexStartA1 != toA1) continue;
        if (((byte)other.Piece & 7) != pieceType) continue;

        int otherFile = other.FromSquare.SquareIndexStartA1 % 8;
        int otherRank = other.FromSquare.SquareIndexStartA1 / 8;
        if (otherFile != fromFile)
        {
          needFile = true;
        }
        else if (otherRank != fromA1 / 8)
        {
          needRank = true;
        }
        else
        {
          needFile = true;
          needRank = true;
        }
      }
      if (needFile) disambig += (char)('a' + fromFile);
      if (needRank) disambig += (char)('1' + fromA1 / 8);
    }

    string prefix = isPawn && isCapture ? ("" + (char)('a' + fromFile)) : "";
    string captureStr = isCapture ? "x" : "";
    string targetStr = "" + (char)('a' + toFile) + (char)('1' + toRank);

    string promoStr = "";
    if (move.PromoteQueen) promoStr = "=Q";
    else if (move.PromoteRook) promoStr = "=R";
    else if (move.PromoteBishop) promoStr = "=B";
    else if (move.PromoteKnight) promoStr = "=N";

    return isPawn
      ? $"{prefix}{captureStr}{targetStr}{promoStr}"
      : $"{pieceStr}{disambig}{captureStr}{targetStr}{promoStr}";
  }


  static void PartialSort(Span<(int idx, double score)> items, int k)
  {
    // Simple selection: move top-k items to front.
    for (int i = 0; i < k; i++)
    {
      int bestIdx = i;
      double bestScore = items[i].score;
      for (int j = i + 1; j < items.Length; j++)
      {
        if (items[j].score > bestScore)
        {
          bestScore = items[j].score;
          bestIdx = j;
        }
      }
      if (bestIdx != i)
      {
        (items[i], items[bestIdx]) = (items[bestIdx], items[i]);
      }
    }
  }
}
