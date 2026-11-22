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
using System.Runtime.Intrinsics.X86;

#endregion

namespace Ceres.MCGS.Utils;

/// <summary>
/// Provides methods for issuing processor prefetch instructions to improve cache performance.
/// </summary>
public static class Prefetcher
{
  public enum CacheLevel
  {
    None,
    Level0,
    Level1,
    Level2
  };

  /// <summary>
  /// Returns true if SSE prefetch instructions are supported on this platform.
  /// </summary>
  public static bool IsPrefetchSupported => Sse.IsSupported;

  /// <summary>
  /// Issues a level 1 prefetch instruction for the specified address.
  /// Only works on x86/x64 platforms. No-op on other platforms.
  /// </summary>
  /// <param name="address"></param>
  public static unsafe void PrefetchLevel1(void* address)
  {
#if DEBUG
    unsafe
    {
      byte _ = *((byte*)address); // would trigger access violation if invalid
    }
#endif

    // Only use prefetch on x86/x64 platforms where SSE is available
    if (Sse.IsSupported)
    {
      // Unclear if prefetch to level 1 or level 2 is faster.
      Sse.Prefetch1(address);
    }
    // No-op on ARM and other platforms
  }


  /// <summary>
  /// Issues a processor prefetch instruction for the specified address and cache level.
  /// Only works on x86/x64 platforms. No-op on other platforms.
  /// </summary>
  /// <param name="address"></param>
  /// <param name="cacheLevel"></param>
  /// <exception cref="Exception"></exception>
  public static unsafe void PrefetchAt(void* address, CacheLevel cacheLevel)
  {
    // Only use prefetch on x86/x64 platforms where SSE is available
    if (!Sse.IsSupported)
    {
      return; // No-op on ARM and other platforms
    }

    switch (cacheLevel)
    {
      case CacheLevel.Level0:
        Sse.Prefetch0(address);
        break;
      case CacheLevel.Level1:
        Sse.Prefetch1(address);
        break;
      case CacheLevel.Level2:
        Sse.Prefetch2(address);
        break;
      case CacheLevel.None:
        break;
      default:
        throw new Exception("Internal error: unsupported cache level");
    }
  }
}

