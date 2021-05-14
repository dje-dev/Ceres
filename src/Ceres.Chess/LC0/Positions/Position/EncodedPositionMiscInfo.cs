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
using System.Diagnostics;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Represents bit layout of a LZ chess position as stored in a 
  /// raw training game file as produced by the LZ client during selfplay mode
  /// (see V3TrainingData structure).
  /// 
  /// 
  /// NOTE: This is a readonly struct.
  ///       It prevents nasty defensive copies that are hard to detect,
  ///       and would break the struct due to the fixed buffers.
  ///       
  ///       Currently the language spec does not allow that for the fixed array fields
  ///       See: https://github.com/dotnet/csharplang/issues/1793
  ///       Consequently we have to write out all the probabilities as their own explicit field
  /// </summary>
  /// 
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public readonly unsafe struct EncodedPositionMiscInfo : IEquatable<EncodedPositionMiscInfo>
  {
    public enum SideToMoveEnum : byte {  White = 0, Black = 1};

    public enum ResultCode : sbyte { Loss = -1, Draw = 0, Win = 1 };

    /// <summary>
    /// If our side has long castling rights.
    /// </summary>
    public readonly byte Castling_US_OOO;

    /// <summary>
    /// If our side has short castling rights.
    /// </summary>
    public readonly byte Castling_US_OO;

    /// <summary>
    /// If other side has long castling rights.
    /// </summary>
    public readonly byte Castling_Them_OOO;

    /// <summary>
    /// If other side has short castling rights.
    /// </summary>
    public readonly byte Castling_Them_OO;


    /// <summary>
    /// Side to move.
    /// </summary>
    public readonly SideToMoveEnum SideToMove; // White = 0, Black = 1

    /// <summary>
    /// Number of moves since 50 move rule reset.
    /// </summary>
    public readonly byte Rule50Count;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="castling_US_OOO"></param>
    /// <param name="castling_US_OO"></param>
    /// <param name="castling_Them_OOO"></param>
    /// <param name="castling_Them_OO"></param>
    /// <param name="sideToMove"></param>
    /// <param name="rule50Count"></param>
    /// <param name="moveCount"></param>
    public EncodedPositionMiscInfo(byte castling_US_OOO, byte castling_US_OO, byte castling_Them_OOO, byte castling_Them_OO, SideToMoveEnum sideToMove, byte rule50Count)
    {
      Castling_US_OOO = castling_US_OOO;
      Castling_US_OO = castling_US_OO;
      Castling_Them_OOO = castling_Them_OOO;
      Castling_Them_OO = castling_Them_OO;
      SideToMove = sideToMove;
      Rule50Count = Math.Min((byte)100, rule50Count); // above 100 LC0 behaves badly (crash or policy does not sum to 100%)
    }


    public override int GetHashCode()
    {
      int part1 = HashCode.Combine(Castling_US_OOO, Castling_US_OO, Castling_Them_OOO, Castling_Them_OO);
      int part2 = HashCode.Combine(SideToMove, Rule50Count);

      return HashCode.Combine(part1, part2);
    }


    /// <summary>
    /// Returns a stable hashcode over the parts of this info
    /// which are used to determine position equality (for neural network evaluation).
    /// Note that the MoveCount is is not used because some nets do not utilize this info (e.g. Leela).
    /// </summary>
    public int HashPosition
    {
      get
      {
        return (Castling_Them_OO << 20) | (Castling_Them_OOO << 19)
              | (Castling_US_OO << 18) | (Castling_US_OOO << 17)
              | ((int)SideToMove << 16) | Rule50Count;
      }
    }


    public override bool Equals(object obj)
    {
      if (obj is EncodedPositionMiscInfo)
        return Equals((EncodedPositionMiscInfo)obj);
      else
        return false;
    }

    public bool Equals(EncodedPositionMiscInfo other)
    {
      return Castling_US_OOO == other.Castling_US_OOO
           && Castling_US_OO == other.Castling_US_OO
           && Castling_Them_OOO == other.Castling_Them_OOO
           && Castling_Them_OO == other.Castling_Them_OO
           && SideToMove == other.SideToMove
           && Rule50Count == other.Rule50Count;
    }

    public bool MirrorEquivalent => Castling_US_OOO   == 0 && Castling_US_OO == 0 
                                 && Castling_Them_OOO == 0 && Castling_Them_OO == 0;

  }
}



#if NOT
  SelfPlayGame::SelfPlayGame(PlayerOptions player1, PlayerOptions player2,
                           bool shared_tree)
    : options_{player1, player2} {
  tree_[0] = std::make_shared<NodeTree>();
  tree_[0]->ResetToPosition(ChessBoard::kStartingFen, {});

  if (shared_tree) {
    tree_[1] = tree_[0];
  } else {
    tree_[1] = std::make_shared<NodeTree>();
    tree_[1]->ResetToPosition(ChessBoard::kStartingFen, {});
  }
}

#endif


