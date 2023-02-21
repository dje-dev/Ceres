// Define this symbol to enable the feature that possibly tracks
// the LastMovePlies in the annotation for each node.
//
// Note that if defined, extra memory/CPU overhead is incurred:
//   - extra space (64 bytes) is always required in all MCTSNodeAnnotation objects, and
//   - if requested by the NNEvaluator, these values will populated during node annotation
//#define LAST_MOVE_PLY_TRACKING

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
using System.Runtime.InteropServices;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

#endregion

namespace Ceres.MCTS.MTCSNodes.Annotation
{
   /// <summary>
   /// Structure which tracks for every square on the board the index of the ply in the game
   /// on which each move last saw action (a piece departing or arriving).
   /// </summary>
  public unsafe struct LastMovePliesTracker
  {
    public const bool DEBUG_DUMP = false;

    // In some cases the true number of ply since last move is:
    //  - unknown (e.g. intialization from a position without history), or
    //  - undefined (e.g. at startpos the pices have technically been infinitely long on their squares).
    // Therefore (by convention) in these situations initialize with a reasonable default,
    // such that the squares don't look to have seen recent activity 
    // but also such that they don't appear to have been there so long they are frozen in some sort of fortress.
    public const int LAST_MOVE_PLY_DEFAULT_VALUE = 30;

#if LAST_MOVE_PLY_TRACKING
    public const bool PlyTrackingFeatureEnabled = true;


    public Span<byte> LastMovePlies => MemoryMarshal.CreateSpan<byte>(ref lastMovePlies[0], 64);

    fixed byte lastMovePlies[64];

    /// <summary>
    /// See description for static method SetMoveSinceFromPositions.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="weAreWhite"></param>
    public void SetMoveSinceFromPositions(Span<Position> positions, bool weAreWhite)
    {
      SetMoveSinceFromPositions(positions, weAreWhite, LastMovePlies);
    }


    /// <summary>
    /// Sets the 64 values in the lastMovePlies array based upon the sequence of positions,
    /// reflecting the number of ply since the first ply of game sequence since each
    /// square has seen a piece move (source or destination square).
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="weAreWhite"></param>
    /// <param name="lastMovePlies"></param>
    /// <exception cref="Exception"></exception>
    public static void SetMoveSinceFromPositions(Span<Position> positions, bool weAreWhite, Span<byte> lastMovePlies)
    {
      lastMovePlies.Clear();
      MGMoveList moves = new MGMoveList();

      for (int s = 0; s < 64; s++)
      {
        lastMovePlies[s] = (byte)Math.Max(LastMovePliesTracker.LAST_MOVE_PLY_DEFAULT_VALUE, positions.Length);
      }

      for (int i = 1; i < positions.Length; i++)
      {
        byte numPliesSinceRootThisPos = (byte)(positions.Length - i - 1);

        Position priorPos = positions[i - 1];
        Position curPos = positions[i];

        // Determine which of the legal moves was played.
        // TODO: this is not highly efficient, consider rewriting if often called.
        moves.Clear();
        MGMoveGen.GenerateMoves(priorPos.ToMGPosition, moves);

        MGMove movePlayed = default;
        for (int m = 0; m < moves.NumMovesUsed; m++)
        {
          MGMove move = moves.MovesArray[m];
          Position newPos = priorPos.AfterMove(MGMoveConverter.ToMove(move));
          if (newPos.PiecesEqual(curPos))
          {
            movePlayed = move;
            break;
          }
        }

        if (movePlayed == default)
        {
          throw new Exception("Internal error: sequence of Positions is illegal at index " + i);
        }

        if (DEBUG_DUMP)
        {
          Console.Write(priorPos + " to " + curPos + " via " + movePlayed + "  ");
        }

        // TOFIX: overflow checking
        if (weAreWhite)
        {
          lastMovePlies[movePlayed.FromSquare.SquareIndexStartH1] = numPliesSinceRootThisPos;
          lastMovePlies[movePlayed.ToSquare.SquareIndexStartH1] = numPliesSinceRootThisPos;
          if (DEBUG_DUMP)
          {
            Console.WriteLine($"update {movePlayed.FromSquare} {movePlayed.FromSquare.SquareIndexStartH1} and {movePlayed.ToSquare} {movePlayed.ToSquare.SquareIndexStartH1}");
          }
        }
        else
        {
          // Reversed, we need final board to be from our perspective which is reversed since we are black.
          lastMovePlies[movePlayed.Reversed.FromSquare.SquareIndexStartH1] = numPliesSinceRootThisPos;
          lastMovePlies[movePlayed.Reversed.ToSquare.SquareIndexStartH1] = numPliesSinceRootThisPos;
          if (DEBUG_DUMP)
          {
            Console.WriteLine($"update {movePlayed.Reversed.FromSquare} {movePlayed.Reversed.FromSquare.SquareIndexStartH1} and {movePlayed.Reversed.ToSquare} {movePlayed.Reversed.ToSquare.SquareIndexStartH1}");
          }
        }
      }
    }


    internal void UpdateLastMovePlyTracking(MCTSNode node, Span<Position> posHistory, bool weAreWhite, in Position Pos, MGMove priorMove)
    {
      if (node.IsRoot)
      {
        SetMoveSinceFromPositions(posHistory, weAreWhite);

        if (LastMovePliesTracker.DEBUG_DUMP)
        {
          Console.WriteLine();
          for (int i = 0; i < 64; i++)
          {
            Console.WriteLine(i + " " + LastMovePlies[i]);
            //          LastMovePlyPerSquare[i] = 30;
          }
        }
      }
      else
      {
        Span<byte> parentLastMovePlyPerSquare = node.Parent.Annotation.LastMovePliesTracker.LastMovePlies;

        // Initialize based on parent values, modified by:
        //   - flipping on the board (change of perspective), and
        //   - increasing by 1 to reflect one level deeper in search tree.
        for (int i = 0; i < 64; i++)
        {
          Square sq = new Square(i);
          LastMovePlies[sq.Reversed.SquareIndexStartH1] = (byte)(parentLastMovePlyPerSquare[i] + 1);
        }

        // Set the source/destiation squares related to this move
        // to have a value of 0 (just moved).
        if (!Pos.IsWhite)
        {
          // Reversed
          LastMovePlies[priorMove.Reversed.FromSquare.SquareIndexStartH1] = 0;
          LastMovePlies[priorMove.Reversed.ToSquare.SquareIndexStartH1] = 0;
        }
        else
        {
          LastMovePlies[priorMove.FromSquare.SquareIndexStartH1] = 0;
          LastMovePlies[priorMove.ToSquare.SquareIndexStartH1] = 0;
        }
      }
    }
#else
    public const bool PlyTrackingFeatureEnabled = false;

    public Span<byte> LastMovePlies => throw new NotImplementedException();

    public void SetMoveSinceFromPositions(Span<Position> positions, bool weAreWhite) => throw new NotImplementedException();

    internal void UpdateLastMovePlyTracking(MCTSNode node, Span<Position> posHistory, bool weAreWhite, in Position Pos, MGMove priorMove)
    {
      // Is a no-op when this feature is not enabled.
    }
#endif

  }

}
