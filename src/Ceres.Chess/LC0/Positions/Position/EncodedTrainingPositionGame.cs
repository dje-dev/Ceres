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
using Ceres.Chess.LC0.Boards;

#endregion


namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Abstract b ase class to encapsulate a game consisting of a sequence of structures 
  /// implementing EncodedTrainingPosition functionality, providing 
  /// history boards, policy, and ancillary training information (e.g. targets).
  /// 
  /// Two subclasses are available:
  ///  - EncodedTrainingPositionGamDirect (positions in original LC0 binary V6 format)
  ///  - EncodedTrainingPositionGameCompressed (in a Ceres packed version of original LC0 binary V6 format)
  ///  
  /// Use of this wrapper allows consumers of training data to be somewhat abstracted away from data representation.
  /// For example, the EncodedTrainingPositionGameCompressed implements policy vectors which are compressed,
  /// reducing memory consumption. Callers will incur the cost of expanding this packed representation
  /// only if and when they actually need to access the policy vector.
  /// </summary>
  public abstract class EncodedTrainingPositionGame
  {
    /// <summary>
    /// Number of positions in sequence.
    /// </summary>
    public abstract int NumPositions { get; }

    /// <summary>
    /// Version number of file.
    /// </summary>
    public abstract int Version { get; }

    /// <summary>
    ///  Board representation input format.
    /// </summary>
    public abstract int InputFormat { get; }

    /// <summary>
    /// Policies (of length 1858 * 4 bytes).
    /// </summary>
    public abstract EncodedPolicyVector PolicyAtIndex(int index);

    /// <summary>
    /// Returns reference to raw (still mirrored) position at a specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    protected abstract ref readonly EncodedPositionWithHistory PositionRawMirroredRefAtIndex(int index);

    #region Base class methods (helpers)

    /// <summary>
    /// Board position (including history planes).
    /// Note that these LC0 training data files (TARs) contain mirrored positions
    /// (compared to the representation that must be fed to the LC0 neural network).
    /// However this mirroring is undone before the boards are returned by this method.
    /// </summary>
    public EncodedPositionWithHistory PositionAtIndex(int index)
    {
      EncodedPositionWithHistory pos = PositionRawMirroredRefAtIndex(index);
      pos.BoardsHistory.MirrorBoardsInPlace();
      return pos;
    }


    /// <summary>
    /// Returns a Memory<EncodedPositionWithHistory>
    /// </summary>
    /// <param name="positionsBuffer"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public Memory<EncodedPositionWithHistory> PositionsAll(EncodedPositionWithHistory[] positionsBuffer)
    {
      if (positionsBuffer.Length < NumPositions)
      {
        throw new Exception("positionsBuffer.Length < NumPositions");
      }
      
      for (int i=0;i<NumPositions;i++)
      {
        positionsBuffer[i] = PositionRawMirroredRefAtIndex(i);
        positionsBuffer[i].BoardsHistory.MirrorBoardsInPlace();  // Undo mirroring  
      }

      return new Memory<EncodedPositionWithHistory>(positionsBuffer, 0, NumPositions);
    }


    static readonly EncodedPositionBoard boardStartPos = EncodedPositionBoard.FromPosition(Position.StartPosition, SideType.White);

    /// <summary>
    /// Returns if the game appears to be a FRC (Fischer random chess) game (does not start with classical starting position).
    /// </summary>
    public bool IsFRCGame
    {
      get
      {
        // Efficient check requiring unmirroring of only first history board.
        EncodedPositionBoard finalBoard = PositionRawMirroredRefAtIndex(0).BoardsHistory.History_0;
        finalBoard.MirrorPlanesInPlace();

        return !finalBoard.Equals(boardStartPos);
      }
    }


    public EncodedPositionMiscInfo PositionMiscInfoAtIndex(int index) => PositionRawMirroredRefAtIndex(index).MiscInfo.InfoPosition;

    public EncodedPositionEvalMiscInfoV6 PositionTrainingInfoAtIndex(int index) => PositionRawMirroredRefAtIndex(index).MiscInfo.InfoTraining;

    public delegate bool MatchBoardPredicate(in EncodedPositionBoard board);

    public bool PosExistsWithCondition(MatchBoardPredicate matchBoardPredicate)
    {
      for (int i = NumPositions - 1; i>= 0; i--)
      {
        ref readonly EncodedPositionBoard firstBoard = ref PositionRawMirroredRefAtIndex(i).BoardsHistory.History_0;
        if (matchBoardPredicate(PositionRawMirroredRefAtIndex(i).BoardsHistory.History_0))
        {
          return true;
        }
      }
      return false;
    }


    /// <summary>
    /// Validates integrity of position at specified index.
    /// </summary>
    /// <param name="index"></param>
    public void ValidateIntegrityAtIndex(int index)
    {
      // Must materialize full EncodedTrainingPosition for this test.
      EncodedTrainingPosition etp = new EncodedTrainingPosition(Version, InputFormat, PositionAtIndex(index), PolicyAtIndex(index));
      etp.ValidateIntegrity("ValidatePositionAtIndex");
    }

    #endregion
  }
}
