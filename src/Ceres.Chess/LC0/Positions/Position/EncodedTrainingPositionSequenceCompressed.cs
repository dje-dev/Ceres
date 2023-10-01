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


#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Wraps an array of EncodedTrainingPosition objects as an EncodedTrainingPositionSequenceBase.
  /// </summary>
  public class EncodedTrainingPositionSequenceCompressed : EncodedTrainingPositionSequenceBase
  {
    EncodedTrainingPositionCompressed[] positions;

    public EncodedTrainingPositionSequenceCompressed(EncodedTrainingPositionCompressed[] positions)
    {
      this.positions = positions ?? throw new ArgumentNullException(nameof(positions));
      if (positions.Length == 0)
      {
        throw new ArgumentException(nameof(positions), "length zero");
      }
    }

    public int NumPositions => positions.Length;

    public override int Version => positions[0].Version;

    public override int InputFormat => positions[0].InputFormat;

    public override EncodedPolicyVector PolicyAtIndex(int index)
    {
      EncodedPolicyVector ret = default;
      EncodedTrainingPositionCompressedConverter.DecompressPolicy(in positions[index].Policies, ref ret);
      return ret;
    }

    public override ref readonly EncodedPositionWithHistory PositionWithBoardsRefAtIndex(int index) => ref positions[index].PositionWithBoards;
  }
}
