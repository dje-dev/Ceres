
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

    /// <summary>
    /// Game result prior to v6.
    /// </summary>
    public readonly byte DepResult;

    /// <summary>
    /// (Win - loss) probability at root. [8280]
    /// </summary>
    public readonly float RootQ;

    /// <summary>
    /// (Win - loss) probability at best move node. [8284]
    /// </summary>
    public readonly float BestQ;

    /// <summary>
    /// Draw probability at root of search. [8288]
    /// </summary>
    public readonly float RootD;

    /// <summary>
    /// Draw probability at best move node. [8292]
    /// </summary>
    public readonly float BestD;

    /// <summary>
    /// MLH average at root at end of search (in plies). [8296]
    /// </summary>
    public readonly float RootM; 

    /// <summary>
    /// MLH average of node having top N at root (in plies). [8300]
    /// </summary>
    public readonly float BestM;

    /// <summary>
    /// Actual number of plies remaining in game. [8304]
    /// </summary>
    public readonly float PliesLeft; 

    /// <summary>
    /// Value head output (W-L) of move with max N. [8308]
    /// </summary>
    public readonly float ResultQ;

    /// <summary>
    /// Value head output (D) of move with max N. [8312]
    /// </summary>
    public readonly float ResultD;

    /// <summary>
    /// Q value of move actually played. [8316]
    /// </summary>
    public readonly float PlayedQ;

    /// <summary>
    /// D value of move actually playeed. [8320]
    /// </summary>
    public readonly float PlayedD;

    /// <summary>
    /// M value of move actually played. [8324]
    /// </summary>
    public readonly float PlayedM;

    // The folowing may be NaN if not found in cache.
    /// <summary>
    /// Value head output (W-L) at root node (possibly NaN if not found in cache, useful for value repair). [8328]
    /// </summary>
    public readonly float OriginalQ;

    /// <summary>
    /// Value head output (D) at root node (possibly NaN if not found in cache, useful for value repair). [8332]
    /// </summary>
    public readonly float OriginalD;

    /// <summary>
    /// Moves left head output (M) at root node (possibly NaN if not found in cache). [8336]
    /// </summary>
    public readonly float OriginalM;

    /// <summary>
    /// Number of visits below root node in search. [8340]
    /// </summary>
    public readonly int NumVisits;

    /// <summary>
    /// Index of the actually played move in game. [8344]
    /// </summary>
    public readonly short PlayedIndex;

    /// <summary>
    /// Index of move having maximal N at end of search. [8346]
    /// </summary>
    public readonly short BestIndex;

    // originally: public readonly long Reserved; [8348]
    public readonly float Unused1;
    public readonly float Unused2;


    #region Helper methods

    /// <summary>
    /// Result (game outcome) WDL distribution.
    /// </summary>
    public (float w, float d, float l) ResultWDL =>  
         (0.5f * (1.0f - ResultD + ResultQ),
          ResultD,
          0.5f * (1.0f - ResultD - ResultQ)
         );
    

    /// <summary>
    /// Neural network WDL distribution at best move node after search.
    /// </summary>
    public (float w, float d, float l) BestWDL =>
         (0.5f * (1.0f - BestD + BestQ),
          BestD,
          0.5f * (1.0f - BestD - BestQ)
         );


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
      throw new NotImplementedException();
      //int part1 = ResultFromOurPerspective.GetHashCode();
      //int part2 = HashCode.Combine(BestD, BestQ, RootD, BestD);

      //return HashCode.Combine(part1, part2);
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
      throw new NotImplementedException();
    }

#endregion
  }
}
