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

using System.Diagnostics;
using System.Runtime.CompilerServices;
using Ceres.Base.Environment;
using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.UserSettings;
using Ceres.MCGS.Graphs.GraphStores;
using Ceres.MCGS.Utils;

#endregion

namespace Ceres.MCGS.Search.Params;

/// <summary>
/// Constant definitions of fixed parameters used to control 
/// various aspects of the MCGS engine.
/// </summary>
public static class MCGSParamsFixed
{
  /// <summary>
  /// If graph rewriting operations should be logged to console.
  /// </summary>
  public const bool GRAPH_REWRITE_DUMP_REUSE_DIAGNOSTICS = true;

  public const bool LOG_LIVE_STATS = false;

  /// <summary>
  /// If true, transposition hash tables use ConcurrentDictionary (legacy behavior).
  /// If false, uses ExtendibleConcurrentHashMap which avoids stop-the-world resize pauses.
  /// </summary>
  public const bool USE_LEGACY_CONCURRENT_DICTIONARY = false;

  public const bool DEBUG_MODE = false;
  public const bool LOGGING_ENABLED = false; // performance degradation high when in Debug mode
  public const bool LOG_ECHO_INLINE = false;

  public const bool VALIDATE_GRAPH_EACH_BATCH = false;

  public const int LOGGING_EXCLUDE_ITERATOR_NUM = -1;

  public const int MIN_N_START_OVERLAP = 1500;


  /// <summary>
  /// Minimum magnitude of change in edge Q to allow propagation of backup
  /// to a parent node not on the visit path.
  /// This limits incurring the performance cost to only situations where there is a
  /// large potential benefit from immediate propagation.
  /// </summary>
  public const double PROPAGATE_OFF_VISIT_PARENTS_MIN_Q_DELTA = 0.01;


  /// <summary>
  /// If the values gathered during the select phase (W and N across all children)
  /// are used to compute and apply an updated Q value to the parent node.
  /// This propagates Q updates to other children (off the current visit path)
  /// that may have happened since the parent node was last visited.
  /// </summary>
  public const bool RESET_Q_DURING_SELECT_PHASE_FROM_ALL_CHILDREN = true;

  public const bool REFRESH_SIBLING_DURING_SELECT_PHASE = true;
  public const bool REFRESH_SIBLING_DURING_BACKUP_PHASE = false;

  public const bool UPDATE_MAXQ_SUBOPTIMALITY = false;

  /// <summary>
  /// Depth in the to which a depth-first attempt is made to
  /// back out draw by repetitions visits which are invalidated
  /// by a change on search root (0 to disable).
  /// 
  /// WARNING: Levels greater than 1 are less well tested.
  /// </summary>
  public const int POSITION_MODE_DEPTH_BACKUP_INVALIDATED_REPETITION = 0;

  /// <summary>
  /// If siblings should choose max(N) for Q.
  /// Tests at 45s+0.75s suggest Elo at least 10 worse when enabled.
  /// </summary>
  public const bool USE_PSEUDOTRANSPOSITION_MAX_N_NODE_ONLY = false;

  public const float SIBLING_POWER_SHRINK_SIBLING_N = 1;
  public const float SIBLING_WT_MAX_FRACTION = 0.65f;

  public const float MOVE_ORDERING_MIN_RATIO_POLICY = 0.15f;

  /// <summary>
  ///  Someday we might also store ActionV at edges (but would grow size of edges).
  /// </summary>
  public const bool GEDGE_HAS_ACTIONV = false;

  /// If the DrawKnownToExistAmongChildren field is updated
  /// and used to adjust backed-up values so that 
  /// any "worse than draw" results are converted to draw results
  /// at the corresponding nodes to reflect more realistic 
  /// minimax optimal evaluation.
  /// </summary>
  public const bool ENABLE_DRAW_KNOWN_TO_EXIST = true; // possibly small benefit? (+3 Elo at 3k nodes)

  public const bool FIX_DRP_NEEDS_3_BEFORE_ROOT = true;
  public const bool FIX_REUSE_GRAPH_DRAW_AT_ROOT = true;

  /// <summary>
  /// If the redescent multiplier should be adjusted higher
  /// for linked transposition nodes with low absolute N.
  /// Two long tests (45s+0.75, 90+1.5 with T3D) suggested circa +10 to +15 Elo benefit.
  /// However a longer test (180+3 with T3D) was only +2 Elo.
  /// </summary>
  public const bool REDESCENT_MUTIPLIER_ADJUST = true;


  // In tests, perhaps especially as N gets larger (e.g. 10000+), numbers less than 0.7 are better (e.g. 0.6 or 0.5)
  public const float RPO_VISIT_COUNT_SHRINK_POWER = 0.6f;//  e.g. 0.5, smaller powers lead to less severe pull toward prior policy
  public const bool RPO_USE_WEIGHTING = false;  // N.B. not yet fully implemented, e.g. in the RPO optimization algorithm

  public const bool VERBOSE_OUTPUT = false;

  public const bool LARGE_HARDWARE_CONFIG = true;

  public const bool TRACK_NODE_EDGE_UNCERTAINTY = false; // methodology is problematic due to aggregated backups

  /// <summary>
  /// Minimum probability threshold for top policy move to be considered for PV auto-extension.
  /// Only moves with policy probability above this threshold will trigger auto-extension.
  /// </summary>
  public const float AUTOEXTEND_MIN_TOP_MOVE_P = 0.5f;

  /// <summary>
  /// If additional integrity assertions (for both Debug and Release builds)
  /// should be enabled (resulting in thrown Exception if failure).
  /// Disabling this for final builds improves performance slightly.
  /// </summary>
#if DEBUG
  public const bool ENABLE_EXTENDED_RELEASE_ASSERTIONS = true;
#else
  public const bool ENABLE_EXTENDED_RELEASE_ASSERTIONS = false;
#endif

  public static void AssertNotNaN(double value)
  {
    bool isOK = !double.IsNaN(value) && !double.IsInfinity(value);
    Assert(isOK, "ValidateNotNaN failed");
  }


  public static void Assert(bool validated, string description)
  {
    if (ENABLE_EXTENDED_RELEASE_ASSERTIONS && !validated)
    {
      WriteCrashDumpFile(description);
      return;
    }
    else
    {
      Debug.Assert(validated, description);
    }
  }


  private static void WriteCrashDumpFile(string description)
  {
    string CRASH_DUMP_FN = $"{description}.dmp";
    CrashDump.WriteDump(@CRASH_DUMP_FN, Microsoft.Diagnostics.NETCore.Client.DumpType.Normal);

    // Dump stack trace to console first
    ConsoleUtils.WriteLineColored(System.ConsoleColor.Red, $"Validation failed {description}");
    ConsoleUtils.WriteLineColored(System.ConsoleColor.Yellow, "Stack trace:");
    ConsoleUtils.WriteLineColored(System.ConsoleColor.Yellow, System.Environment.StackTrace);
    
    ConsoleUtils.WriteLineColored(System.ConsoleColor.Red, $"Dumping crash dump to "
      + $"{System.Environment.CurrentDirectory} {CRASH_DUMP_FN}...");

    Debugger.Break();
    throw new System.Exception("Internal error: " + description);
  }

  /// <summary>
  /// If operating system large pages (2MB each under Windows) 
  /// should be used for the arrays of raw nodes ahd child infos.
  /// 
  /// This potentially reduces memory access time, but requires
  /// elevated priveleges and sufficient contiguous memory available 
  /// (which cannot be paged)
  /// 
  /// NOTE: On a dual socket machine performance was clearly inferior with large pages.
  ///       On a single socket machine performance was considerably improved, although
  ///       a limitation (on Windows only) is that larges pages is incompatible 
  ///       with incremental allocation under Windows.
  ///       
  /// NOTE: On Linux sometimes large page allocations may fail and seemingly cannot 
  ///       be detected in the mreserve or mprotect call, causing access violations.
  /// </summary>
  public static bool TryEnableLargePages => CeresUserSettingsManager.Settings.UseLargePages && SoftwareManager.IsLinux;

  /// <summary>
  /// In incremental storage mode memory is reserved at initialization
  /// but only actually committed incrementally as the search tree grows.
  /// </summary>
  public const bool STORAGE_USE_INCREMENTAL_ALLOC = GraphStoreConfig.STORAGE_USE_INCREMENTAL_ALLOC;

  public const int PARALLEL_SELECT_NUM_INITIAL_WORKERS = 4;
  public const int PARALLEL_SELECT_NUM_WORKERS_GROWTH_INCREMENT = 2;


  // **************************************
  //  NOTE: 
  // These shadowed in GraphStoreConfig
  // **************************************
  /// <summary>
  /// The use of int data type to encode N value within nodes limits 
  /// the number of possible visits within a tree.
  /// Furthermore, some data structures need to fit in an int
  /// (such as child edge block indices). 
  /// Although number of nodes may be much less than number of visits,
  /// exact guarantees are not available.
  /// Thus we use a value of about 1 billion currently.
  /// </summary>
  public const int MAX_VISITS = (int.MaxValue / 2) - 1_000_000;

  /// <summary>
  /// Type of prefetching to be used (if any) at certain points in search
  /// where it can be predicted which next data item will be needed.
  /// 
  /// Significant performance benefit is typically seem for larger searches (circa 10% speedup).
  /// </summary>
  public const Prefetcher.CacheLevel PrefetchCacheLevel = Prefetcher.CacheLevel.Level1;
}
