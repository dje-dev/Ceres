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
using System.IO;
using System.Runtime.InteropServices;
using Ceres.Base.Misc;
using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.Chess.NNEvaluators.LC0DLL
{
  /// <summary>
  /// Interface to Syzygy tablebase probing routines 
  /// exposed by the LC0 DLL.
  /// </summary>
  public partial class LC0DLLSyzygyEvaluator : IDisposable, ISyzygyEvaluatorEngine
  {
    public enum WDLScore
    {
      /// <summary>
      /// Loss
      /// </summary>
      WDLLoss = -2,

      /// <summary>
      /// Loss, but draw under 50-move rule
      /// </summary>
      WDLBlessedLoss = -1,

      /// <summary>
      /// Draw
      /// </summary>
      WDLDraw = 0,

      /// <summary>
      /// Win, but draw under 50-move rule
      /// </summary>
      WDLCursedWin = 1,

      /// <summary>
      /// Win
      /// </summary>
      WDLWin = 2
    }
    public enum ProbeState
    {
      /// <summary>
      /// DTZ should check other side
      /// </summary>
      ChangeSTM = -1,

      /// <summary>
      /// Fail
      /// </summary>
      Fail = 0,

      /// <summary>
      /// Ok
      /// </summary>
      Ok = 1,

      /// <summary>
      /// Best move zeros DTZ
      /// </summary>
      ZeroingBestMove = 2
    }


    /// <summary>
    /// Static counter of succesful tablebase probes.
    /// </summary>
    public static long NumTablebaseHits = 0;

    /// <summary>
    /// Session ID under which the registration with LC0 DLL was made.
    /// </summary>
    private int sessionID;

    /// <summary>
    /// Returns the maximum cardinality supported by loaded tablebase(number of pieces on board).
    /// </summary>
    public int MaxCardinality { private set; get; }


    /// <summary>
    /// Constructor that prepares evaluator.
    /// </summary>
    /// <param name="sessionID"></param>
    public LC0DLLSyzygyEvaluator(int sessionID)
    {
      this.sessionID = sessionID;
    }

    public bool DTZAvailable { private set; get; }

    /// <summary>
    /// Internal initialization routine to register with LC0 DLL.
    /// </summary>
    /// <param name="paths"></param>
    /// <returns></returns>
    public bool Initialize(string paths)
    {
      // Validate that all the requested paths actually exist
      if (!FileUtils.PathsListAllExist(paths))
      {
        throw new Exception($"One or more specified Syzygy paths not found or inaccessible: {paths} ");
      }

      LCO_Interop.CheckLibraryDirectoryOnPath();

      // Determine the maximum cardinality (number of pieces) supported
      LCO_Interop.TBInitializeStatus initStatus = LCO_Interop.TBInitialize(sessionID, paths);
      if (initStatus == LCO_Interop.TBInitializeStatus.ERROR)
      {
        throw new Exception($"Loading tablebases failed, attempted {paths}");
      }
      else
      {
        DTZAvailable = initStatus == LCO_Interop.TBInitializeStatus.OK_WITH_DTM_DTZ;
        MaxCardinality = LCO_Interop.MaxCardinality(sessionID);
        return true;
      }
    }

    /// <summary>  
    /// Probes WDL tables for the given position to determine a WDLScore.
    /// Result is only strictly valid for positions with 0 ply 50 move counter.
    ///</ summary>
    public void ProbeWDL(in Position pos, out WDLScore score, out ProbeState result)
    {
      // Make sure sufficiently few pieces remain on board
      // and not castling rights
      if (pos.PieceCount > MaxCardinality || pos.MiscInfo.CastlingRightsAny)
      {
        score = default;
        result = ProbeState.Fail;
        return;
      }

      // Call DLL function to do the probe and return encoded result
      int resultCode = LCO_Interop.ProbeWDL(sessionID, pos.FEN);

      // Unpack the encoded result code
      result = (ProbeState)(resultCode / 256 - 10);
      score = (WDLScore)(resultCode % 256 - 10);

      if (result == ProbeState.Ok || result == ProbeState.ZeroingBestMove)
      {
        NumTablebaseHits++;
      }
    }


    /// <summary>/
    /// Probes DTZ tables for the given position to determine the number of ply
    /// before a zeroing move under optimal play.
    ///</summary>
    public int ProbeDTZ(in Position pos)
    {
      // Make sure sufficiently few pieces remain on board
      // and not castling rights
      if (!DTZAvailable 
       || pos.PieceCount > MaxCardinality 
       || pos.MiscInfo.CastlingRightsAny)
      {
        return -1;
      }

      NumTablebaseHits++;

      // Call DLL function to do the probe and return encoded result
      return LCO_Interop.ProbeDTZ(sessionID, pos.FEN);
    }

    /// Probes DTZ tables to determine which moves are on the optimal play path.
    /// Assumes the position is one reached such that the side to move has been
    /// performing optimal play moves since the last 50 move counter reset.
    /// has_repeated should be whether there are any repeats since last 50 move
    /// counter reset.
    /// Safe moves are added to the safe_moves output paramater.

    public bool RootProbe(string position, bool hasRepeated, List<MGMove> moves)
    {
      //bool root_probe(const Position&pos, bool has_repeated, std::vector < Move > *safe_moves);
      return false;
    }

    /// <summary>
    /// Probes WDL tables to determine which moves might be on the optimal play
    /// path. If 50 move ply counter is non-zero some (or maybe even all) of the
    /// returned safe moves in a 'winning' position, may actually be draws.
    /// Safe moves are added to the safe_moves output paramater.
    /// </summary>
    public bool RootProbeWDL(string position, List<MGMove> moves)
    {
      //  bool root_probe_wdl(const Position& pos, std::vector<Move>* safe_moves);
      return false;
    }

#region Disposal

    bool isDisposed = false;

    ~LC0DLLSyzygyEvaluator()
    {
      Dispose();
    }

    
    public void Dispose()
    {
      if (!isDisposed)
      {
        // Currently we do nothing.
      }

      isDisposed = true;
    }

#endregion

  }
}
