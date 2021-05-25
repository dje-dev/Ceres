
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

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Miscellaneous information associated with a position appearing in training data
  /// (binary compatible with LZ training files).
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public readonly struct EncodedPositionEvalMiscInfoV6: IEquatable<EncodedPositionEvalMiscInfoV6>
  {
    // Bitfield with the following allocation:
    //  bit 7: side to move (input type 3)
    //  bit 6: position marked for deletion by the rescorer (never set by lc0)
    //  bit 5: game adjudicated (v6)
    //  bit 4: max game length exceeded (v6)
    //  bit 3: best_q is for proven best move (v6)
    //  bit 2: transpose transform (input type 3)
    //  bit 1: mirror transform (input type 3)
    //  bit 0: flip transform (input type 3)
    public readonly byte InvarianceInfo;
    public readonly EncodedPositionMiscInfo.ResultCode ResultFromOurPerspective;

    public readonly float RootQ;
    public readonly float BestQ;
    public readonly float RootD;
    public readonly float BestD;

    public readonly float RootM;      // In plies.             
    public readonly float BestM;      // In plies.              
    public readonly float PliesLeft;

    public readonly float ResultQ;
    public readonly float ResultD;
    public readonly float PlayedQ;
    public readonly float PlayedD;
    public readonly float PlayedM;

    // The folowing may be NaN if not found in cache.
    public readonly float OriginalQ;      // For value repair.     
    public readonly float OriginalD;
    public readonly float OriginalM;
    public readonly int NumVisits;

    // Indices in the probabilities array.
    public readonly short PlayedIndex;
    public readonly short BestIndex;

    // originally: public readonly long Reserved;
    public readonly float Unused1;
    public readonly float Unused2;


    #region Helper methods

    /// <summary>
    /// Returns if the played move was not the best move (noise induced alternate chosen).
    /// </summary>
    public bool NotBestMove => PlayedIndex != BestIndex;

    /// <summary>
    /// Returns the Q amount by which the playing the best move would have 
    /// been better then the move actually played.
    /// </summary>
    public float QSuboptimality => BestQ - PlayedQ;

    /// <summary>
    /// Fill-in value for the small (circa 3%) fraction of source data 
    /// with missing (NaN) OriginalQ data.
    /// </summary>
    public const float FILL_IN_UNCERTAINTY = 0.15f;


    /// <summary>
    ///  Uncertainty is a measure of how incorrect the position evaluation (V)
    ///  turned out to be relative to the search evaluation (Q).
    /// </summary>
    public float Uncertainty => float.IsNaN(OriginalQ) ? FILL_IN_UNCERTAINTY : MathF.Pow(OriginalQ - BestQ, 2);


    public override int GetHashCode()
    {
      int part1 = ResultFromOurPerspective.GetHashCode();
      int part2 = HashCode.Combine(BestD, BestQ, RootD, BestD);

      return HashCode.Combine(part1, part2);
    }


    public override bool Equals(object obj)
    {
      if (obj is EncodedPositionEvalMiscInfoV6)
        return Equals((EncodedPositionEvalMiscInfoV6)obj);
      else
        return false;
    }


    public bool Equals(EncodedPositionEvalMiscInfoV6 other)
    {
      return this.ResultFromOurPerspective == other.ResultFromOurPerspective
           && this.RootQ == other.RootQ
           && this.BestQ == other.BestQ
           && this.RootD == other.RootD
           && this.BestD == other.BestD;
    }

    #endregion
  }
}
