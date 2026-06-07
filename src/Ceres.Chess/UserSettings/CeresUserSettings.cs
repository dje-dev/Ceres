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

#endregion

namespace Ceres.Chess.UserSettings
{
  /// <summary>
  /// Set of Ceres configuration settings specified by user
  /// and persisted in cofiguration file.
  /// </summary>
  public record CeresUserSettings
  {
    /// <summary>
    /// Directory in which LC0 binaries (executable and/or library) are located.
    /// </summary>
    public string DirLC0Binaries { get; set; } = ".";

    /// <summary>
    /// Name of the LC0 executable (e.g. the default of lc0").
    /// </summary>
    public string LC0ExeName { get; set; } = "lc0";

    /// <summary>
    /// Directory in which LC0 network weights files are located
    /// </summary>
    public string DirLC0Networks { get; set; } = ".";

    /// <summary>
    /// Directory in which Ceres network weights files are located
    /// </summary>
    public string DirCeresNetworks { get; set; } = ".";

    /// <summary>
    /// Directory in which cached TensorRT engine files are cached.
    /// Default value of null causes a system-created temporary directory to be used.
    /// </summary>
    public string DirTRTEngines { get; set; } = null;


    /// <summary>
    /// Directory in which EPD files are located.
    /// </summary>
    public string DirEPD { get; set; } = ".";

    /// <summary>
    /// Directory in which PGN files are located.
    /// </summary>
    public string DirPGN { get; set; } = ".";

    /// <summary>
    /// Directory in which output files from Ceres (e.g. PGN) are located.
    /// </summary>
    public string DirCeresOutput { get; set; } = ".";

    /// <summary>
    /// Directory in which external UCI engines (executables) are found.
    /// </summary>
    public string DirExternalEngines { get; set; } = ".";

    /// <summary>
    /// Default value for Ceres network specification string (for network evaluation).
    /// </summary>
    public string DefaultNetworkSpecString { get; set; }

    /// <summary>
    /// Directory containing Graphviz binaries (executables).
    /// </summary>
    public string DirGraphvizBinaries { get; set; } = null;

    /// <summary>
    /// Defalut value for Ceres device specification string (for network evaluation).
    /// </summary>
    public string DefaultDeviceSpecString { get; set; }

    /// <summary>
    /// Base URL to download location of LC0 weights files (e.g. http://training.lczero.org/networks).
    /// </summary>
    public string URLLC0Networks { get; set; } = @"http://training.lczero.org/networks";

    /// <summary>
    /// Ceres will not run if compiled in Debug mode
    /// (Release required) unless this setting is true
    /// or alternately the OS environment variable CERES_DEBUG exists.
    /// </summary>
    public bool DebugAllowed { get; set; } = false;

    #region Reference engine

    /// <summary>
    /// Full path and name of executable used as a "reference" engine (if any).
    /// </summary>
    public string RefEngineEXE { get; set; } = "stockfish.exe";

    /// <summary>
    /// Size of hash table allocated for reference engine (in megabytes).
    /// </summary>
    public int RefEngineHashMB { get; set; } = 2048;

    /// <summary>
    /// Number of threads allocated to reference engine.
    /// </summary>
    public int RefEngineThreads { get; set; } = 8;

    /// <summary>
    /// Full path to neural network file to be used with reference engine
    /// (if it happens to be LC0 or Ceres).
    /// </summary>
    public string RefEngineNetworkFile { get; set; } = null;

    #endregion

    #region Metrics and Logging

    /// <summary>
    /// If the .NET metrics monitoring window should be launced at startup.
    /// </summary>
    public bool LaunchMonitor { get; set; } = false;

    /// <summary>
    /// If logging messages of Info severity should be output.
    /// </summary>
    public bool LogInfo { get; set; } = false;

    /// <summary>
    /// If logging messages of Warn severity should be output.
    /// </summary>
    public bool LogWarn { get; set; } = false;

    /// <summary>
    /// If memory allocations should attempt to use large pages for improved performance.
    /// </summary>
    public bool UseLargePages { get; set; } = false;

    /// <summary>
    /// The maximum number tree nodes allowed in any single search tree
    /// (yielding a number of visits which is typically 10% to 50% greater).
    /// The average number of bytes per tree nodes is approximately 200.
    /// If not explicitly set then a value will be automatically 
    /// set based on the physical memory in the computer.
    /// </summary>
    public int? MaxTreeNodes { get; set; } = null;

    /// <summary>
    /// The maximum number of positions generated per leaf selection batch.
    /// Higher values improve CPU througput by allowing more parallelism.
    /// Lower values improve the purity of selected nodes, which can
    /// significantly improve play quality especially for larger, accurate networks.
    /// </summary>
    public int MaxBatchSize { get; set; } = 1024;

    /// <summary>
    /// If the CUDA graphs features should be used 
    /// improve speed of execution of smaller batches
    /// (at the expense of higher GPU memory utilization).
    /// </summary>
    public bool EnableCUDAGraphs { get; set; } = true;

    /// <summary>
    /// If the "dual overlapped executors" feature should be enabled
    /// for possible use to run two concurrent executors
    /// (in different CUDA streams).
    /// Disabling saves some GPU memory at the cost of some reduction in search speed.
    /// </summary>
    public bool EnableOverlappingExecutors { get; set; } = true;

    /// <summary>
    /// If data structures and algorithms should favor low
    /// memory consumption (at a cost of modest performance reduction).
    /// </summary>
    public bool ReducedMemoryMode { get; set; } = false;

    /// <summary>
    /// Processor NUMA node to which engine should be affinitized.
    /// Generally a single node (socket or possibly chiplet) should be 
    /// selected to improve performance.
    /// Value of -1 indicates that affinitization should be disabled.
    /// </summary>
    public int NUMANode { get; set; } = 0;


    #endregion

    #region UCI setoptions

    public string UCILogFile { get; set; }
    public string SearchLogFile { get; set; }
    public bool VerboseMoveStats { get; set; } = false;
    public float? SmartPruningFactor { get; set; }

    public float? CPUCT { get; set; }
    public float? CPUCTAtRoot { get; set; }
    public float? CPUCTBase { get; set; }
    public float? CPUCTBaseAtRoot { get; set; }
    public float? CPUCTFactor { get; set; }
    public float? CPUCTFactorAtRoot { get; set; }
    public float? PolicyTemperature { get; set; }
    public float? ValueTemperature { get; set; }
    public float? FPU { get; set; }
    public float? FPUAtRoot { get; set; }

    public bool? EnableSiblingEval { get; set; }
    public bool? EnableUncertaintyBoosting { get; set; }
    public string LimitsManagerName { get; set; }

    #region Tablebases

    public string TablebaseDirectory
    {
      // Prefer SyzygyPath entry but also support DirTablebases
      // for legacy compatability.
      get
      {
        bool hasDirTablebases = dirTablebases != null & dirTablebases != "";
        bool hasSyzygyPath = SyzygyPath != null & SyzygyPath != "";
        if (hasDirTablebases && hasSyzygyPath)
        {
          throw new Exception("Ceres.json cannot contain both SyzygyPath and DirTablebases");
        }

        if (hasDirTablebases)
          return DirTablebases;
        else if (hasSyzygyPath)
          return SyzygyPath;
        else
          return null;
      }
    }

    string dirTablebases;

    /// <summary>
    /// Directory in which Syzygy tablebases are located, or null if none
    /// (synonym for SyzygyPath).
    /// </summary>
    public string DirTablebases { get { return dirTablebases ?? SyzygyPath; } set { dirTablebases = value; } }


    /// <summary>
    /// Directory in which Syzygy tablebases are located, or null if none
    /// (synonym for DirTablebases).
    /// </summary>
    public string SyzygyPath { get; set; }

    #endregion

    #endregion
  }

}
