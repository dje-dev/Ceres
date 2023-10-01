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



#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Base class to encapsulate a sequence of EncodedTrainingPosition struct
  /// </summary>
  public abstract class EncodedTrainingPositionSequenceBase
  {
    /// <summary>
    /// Number of positions in sequence.
    /// </summary>
    public int NumPosition { get; }

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
    /// Board position (including history planes).
    /// Note that these LC0 training data files (TARs) contain mirrored positions
    /// (compared to the representation that must be fed to the LC0 neural network).
    /// However this mirroring is undone immediately after reading from disk and
    /// this field in memory is always in the natural representation.
    /// </summary>
    public abstract ref readonly EncodedPositionWithHistory PositionWithBoardsRefAtIndex(int index);
  }
}
