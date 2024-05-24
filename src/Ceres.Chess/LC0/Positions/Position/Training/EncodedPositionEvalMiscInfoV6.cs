
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
using Ceres.Chess.EncodedPositions.Basic;

#endregion

namespace Ceres.Chess.EncodedPositions
{
  /// <summary>
  /// Miscellaneous information associated with a position appearing in training data
  /// (binary compatible with LZ training files).
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  public readonly record struct EncodedPositionEvalMiscInfoV6: IEquatable<EncodedPositionEvalMiscInfoV6>
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

    /// <summary>
    /// Kullback-Leibler divergence between policy head and visits in nats (originally was unused) [8348] 
    /// NOTE: Unfortunately this value was accidentally overwritten in the EncodedTrainingPositionCompressed format
    ///       but only on the first position of each game. The accessor below works around this.
    /// TODO: Someday fix this, but it will require rewriting all files. When we do this:
    ///         - instead use the Unused2 field to store these sentinels
    ///         - remove the accessor below, make it a readonly field just like the others
    ///         - modify the methods at the top of the EncodedTrainingPositionCompressedConverter class to reflect this
    /// </summary>
    internal readonly float KLDPolicyRawOrSentinel;

    public readonly float KLDPolicy
    {
      get
      {
        if (KLDPolicyRawOrSentinel == EncodedTrainingPositionCompressedConverter.SENTINEL_MARK_FIRST_MOVE_IN_GAME_IN_UNUSED1)
        {
          return 0.02f; // A dummy value, representative of what KLD would typically be for the first position in a game.
        }
        else
        {
          return KLDPolicyRawOrSentinel;
        }
      }
    }

    public readonly float Unused2; // [8350]


    #region Helper methods

    /// <summary>
    /// Value head output (WDL) at root node .
    /// </summary>
    public (float w, float d, float l) OriginalWDL
    {
      get
      {
        if (float.IsNaN(OriginalD+OriginalQ))
        {
          return (float.NaN, float.NaN, float.NaN);
        }

        float l = (OriginalQ - 1 + OriginalD) / -2;
        float w = 1 - OriginalD - l;
        Debug.Assert(Math.Abs(w + l + OriginalD - 1) < 0.0001f);
        Debug.Assert(Math.Abs((w - l) - OriginalQ) < 0.0001f);

        return (w, OriginalD, l);
      }
    }


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
    public (float w, float d, float l) BestWDL
    {
      get
      {
        float sum = BestQ + BestD;
        if (float.IsNaN(sum) || sum == 0)
        {
          // BestWDL was not available, fall back to ResultWDL.
          return ResultWDL;
        }
        else
        {
          return (0.5f * (1.0f - BestD + BestQ),
            BestD,
            0.5f * (1.0f - BestD - BestQ)
           );
        }
      }
    }


    /// <summary>
    /// Returns the move actually played.
    /// 
    /// NOTE: It is recommended to use PlayedMove property at the parent structure (EncodedTrainingPosition.PlayedMove)
    ///       to avoid having deal with the internal details of the encoding (e.g .mirroring status).
    /// </summary>
    public readonly EncodedMove PlayedMove => EncodedMove.FromNeuralNetIndex(PlayedIndex);


    /// <summary>
    /// Returns the best move according to search.
    /// 
    /// NOTE: It is recommended to use BestMove property at the parent structure (EncodedTrainingPosition.BestMove)
    ///       to avoid having to deal with the internal details of the encoding (e.g .mirroring status).
    /// </summary>
    public readonly EncodedMove BestMove => EncodedMove.FromNeuralNetIndex(BestIndex);


    /// <summary>
    /// Overwrites the value of PlayedIndex and BestIndex.
    /// WARNING: is unsafe.
    /// </summary>
    /// <param name="miscInfo"></param>
    public readonly unsafe void SetPlayedAndBestIndex(short playedIndex, short bestIndex)
    {
      fixed (short* p = &PlayedIndex)
      {
        *p = playedIndex;
      }
      fixed (short* p = &BestIndex)
      {
        *p = bestIndex;
      }
    }

    /// <summary>
    /// Overwrites the value of OriginalQ, OriginalD, OriginalM.
    /// WARNING: is unsafe.
    /// </summary>
    public readonly unsafe void SetOriginal(float originalQ, float originalD, float originalM)
    {
      fixed (float* p = &OriginalQ)
      {
        *p = originalQ;
      }
      fixed (float* p = &OriginalD)
      {
        *p = originalD;
      }
      fixed (float* p = &OriginalM)
      {
        *p = originalM;
      }
    }

    /// <summary>
    /// Overwrites the value of KLDPolicy.
    /// WARNING: is unsafe.
    /// </summary>
    /// <param name="value"></param>
    internal unsafe void SetKLDPolicy(float value)
    {
      fixed (float* p = &KLDPolicyRawOrSentinel)
      {
        *p = value;
      }
    }

    /// <summary>
    /// Overwrites the value of Unused2.
    /// WARNING: is unsafe.
    /// </summary>
    /// <param name="value"></param>
    internal unsafe void SetUnused2(float value)
    {
      fixed (float* p = &Unused2)
      {
        *p = value;
      }
    }


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

    public EncodedPositionEvalMiscInfoV6(byte invarianceInfo, byte depResult, float rootQ, float bestQ, float rootD, 
                                         float bestD, float rootM, float bestM, 
                                         float pliesLeft, 
                                         float resultQ, float resultD,
                                         float playedQ, float playedD, float playedM, 
                                         float originalQ, float originalD, float originalM, 
                                         int numVisits, short playedIndex, short bestIndex, 
                                         float unused1, float unused2)
    {
      InvarianceInfo = invarianceInfo;
      DepResult = depResult;
      RootQ = rootQ;
      BestQ = bestQ;
      RootD = rootD;
      BestD = bestD;
      RootM = rootM;
      BestM = bestM;
      PliesLeft = pliesLeft;
      ResultQ = resultQ;
      ResultD = resultD;
      PlayedQ = playedQ;
      PlayedD = playedD;
      PlayedM = playedM;
      OriginalQ = originalQ;
      OriginalD = originalD;
      OriginalM = originalM;
      NumVisits = numVisits;
      PlayedIndex = playedIndex;
      BestIndex = bestIndex;
      Unused2 = unused1;
      Unused2 = unused2;
    }


    /// <summary>
    ///  Uncertainty is a measure of how incorrect the position evaluation (V)
    ///  turned out to be relative to the search evaluation (Q).
    /// </summary>
    public float Uncertainty => float.IsNaN(OriginalQ + BestQ) ? FILL_IN_UNCERTAINTY : MathF.Abs(OriginalQ - BestQ);

    #endregion
  }
}
