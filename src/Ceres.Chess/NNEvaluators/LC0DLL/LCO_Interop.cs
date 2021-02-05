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
using System.IO;
using System.Runtime.InteropServices;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.Chess.NNEvaluators.LC0DLL
{
  /// <summary>
  /// Defines the external functions and data structures used to
  /// communicate with the custom LC0 library  with
  /// backend logic for neural network evaluation or Syzygy probes.
  /// </summary>
  internal static class LCO_Interop
  {
    /// <summary>
    /// Base filename of the custom library (.dll or .so).
    /// </summary>
    const string LC0_LIBRARY_FILENAME = "LC0";

    /// <summary>
    /// Maximum number of session that the library can support 
    /// simultaneously (this limit hardcoded in C++).
    /// </summary>
    internal const int MAX_SESSIONS = 32;

    /// <summary>
    /// The LZ binary is complied with a maximum batch size, typically 1024
    /// </summary>
    internal const int MAX_POSITIONS_PER_BATCH = 1024;

    #region Evaluator functions    

    public enum TBInitializeStatus : int
    {
      ERROR = 0,
      OK_WITH_DTM = 1,
      OK_WITH_DTM_DTZ = 2
    }

    public enum AllocErrorCode : int
    {
      NO_ERROR = 0,
      ERROR_INVALID_GPU_ID = -2
    }

    [DllImport(LC0_LIBRARY_FILENAME)]
    internal static extern AllocErrorCode Alloc(int sessionIndex, string networkFilename, int gpuID);
    [DllImport(LC0_LIBRARY_FILENAME)]
    internal static extern void Free(int sessionIndex);

    [DllImport(LC0_LIBRARY_FILENAME)]
    internal unsafe static extern int Compute(int sessionIndex, int batchSize,
                                              CeresTransferBlockInItem* inputs,
                                              CeresTransferBlockOutItem* outputs);
    #endregion


    #region Tablebase functions

    [DllImport(LC0_LIBRARY_FILENAME)]
    internal static extern TBInitializeStatus TBInitialize(int sessionIndex, string paths);

    [DllImport(LC0_LIBRARY_FILENAME)]
    internal static extern int TBFree(int sessionIndex);

    [DllImport(LC0_LIBRARY_FILENAME)]
    internal static extern int ProbeWDL(int sessionIndex, string fen);

    [DllImport(LC0_LIBRARY_FILENAME)]
    internal static extern int ProbeDTZ(int sessionIndex, string fen);


    [DllImport(LC0_LIBRARY_FILENAME)]
    internal static extern int MaxCardinality(int sessionIndex);

    #endregion

    #region Transfer structures

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    internal unsafe struct CeresTransferBlockInItem
    {
      // Fields used if neural network evaluation requested
      internal fixed ulong Masks[CeresTransferBlockIn.NUM_PLANES_PER_POSITION];
      internal fixed float Values[CeresTransferBlockIn.NUM_PLANES_PER_POSITION];
      internal ulong PositionHash;
      internal int NumMoves;
      internal fixed short Moves[CeresTransferBlockIn.MAX_MOVES];
    };

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    internal unsafe struct CeresTransferBlockIn
    {
      /// <summary>
      /// For efficiency (space saving) reasons, the maximum number
      /// of possible moves is limited at this value.
      /// </summary>
      internal const int MAX_MOVES = 96;
      internal const int NUM_PLANES_PER_POSITION = 112;

      const int CERES_INPUT_PLANE_SIZE_NUM_ELEMENTS = NUM_PLANES_PER_POSITION * MAX_POSITIONS_PER_BATCH;
      const int CERES_INPUT_PLANE_SIZE_BYTES = 12;
    };

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    internal unsafe struct CeresTransferBlockOutItem
    {
      internal float Q;
      internal float D;
      internal fixed float P[CeresTransferBlockIn.MAX_MOVES];
      internal float M;
    }

    #endregion


    #region Initialization

    static bool haveCheckedDLLLoad = false;
    static readonly object checkDLLLoadLockObj = new object();

    /// <summary>
    /// Prepends the path specified in the user setting DirLC0Binaries
    /// to the path so that the library (LC0.DLL or LC0.so) will
    /// be found by the operating system when we try to load it.
    /// 
    /// It is expected that the LC0 binaries directory will generally contain
    /// a fulll distribution LC0 including the executable and all 
    /// associated libraries (such as CUDA).
    /// </summary>
    internal static void CheckLibraryDirectoryOnPath()
    {
      if (!haveCheckedDLLLoad)
      {
        lock (checkDLLLoadLockObj)
        {
          if (!haveCheckedDLLLoad)
          {
            // Verify the path exists
            string lc0Path = CeresUserSettingsManager.Settings.DirLC0Binaries;
            if (!Directory.Exists(lc0Path))
            {
              throw new Exception("The path specified in user setting DirLC0Binaries does not exist");
            }
            else
            {
              // Prepend the path to this binary 
              string currentPath = Environment.GetEnvironmentVariable("PATH");
              Environment.SetEnvironmentVariable("PATH", lc0Path + Path.PathSeparator + currentPath);

              haveCheckedDLLLoad = true;
            }
          }
        }
      }
    }
    #endregion

  }
}
