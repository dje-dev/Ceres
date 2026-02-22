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

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// TensorRT runtime manager. Handles initialization and shutdown of the TensorRT library.
/// </summary>
public sealed class TensorRT : IDisposable
{
  /// <summary>
  /// Lazily-initialized singleton instance of TensorRT.
  /// </summary>
  private static readonly Lazy<TensorRT> lazyInstance = new(() => new TensorRT());

  /// <summary>
  /// Gets the shared singleton instance of TensorRT.
  /// </summary>
  public static TensorRT Instance => lazyInstance.Value;

  private bool initialized;
  private bool disposed;


  /// <summary>
  /// Constructor. Initializes the TensorRT runtime.
  /// </summary>
  public TensorRT()
  {
    int result = TensorRTNative.Init();
    if (result != 0)
    {
      string error = TensorRTNative.GetLastErrorString();
      throw new InvalidOperationException($"TensorRT init failed ({result}): {error ?? "unknown error"}");
    }
    initialized = true;
  }


  /// <summary>
  /// Gets the TensorRT version as (Major, Minor, Patch).
  /// </summary>
  public static (int Major, int Minor, int Patch) Version
  {
    get
    {
      int version = TensorRTNative.GetVersion();
      int major = version / 10000;
      int minor = (version / 100) % 100;
      int patch = version % 100;
      return (major, minor, patch);
    }
  }


  /// <summary>
  /// Load engine from ONNX file (no caching).
  /// </summary>
  public TensorRTEngine LoadEngine(string onnxPath, int batchSize, TensorRTBuildOptions? options = null, int deviceId = -1)
  {
    return new TensorRTEngine(onnxPath, batchSize, options, deviceId);
  }


  /// <summary>
  /// Load engine with caching support. Checks cache first, builds if needed.
  /// </summary>
  public TensorRTEngine LoadEngineWithCache(string onnxPath, int batchSize, TensorRTBuildOptions options,
                                             string cacheDir, int deviceId = -1, bool forceRebuild = false)
  {
    return TensorRTEngine.LoadWithCache(onnxPath, batchSize, options, deviceId, cacheDir, forceRebuild);
  }


  /// <summary>
  /// Build a single multi-profile engine with shared weights,
  /// returning one TensorRTEngine per batch size.
  /// </summary>
  public TensorRTEngine[] LoadMultiProfileEngineWithCache(string onnxPath, int[] batchSizes,
      TensorRTBuildOptions options, string cacheDir, int deviceId = -1, bool forceRebuild = false)
  {
    return TensorRTEngine.LoadMultiProfileWithCache(onnxPath, batchSizes, options, deviceId, cacheDir, forceRebuild);
  }


  /// <summary>
  /// Load a pre-built multi-profile engine file (.engine) directly,
  /// bypassing ONNX parsing and cache validation.
  /// </summary>
  public TensorRTEngine[] LoadMultiProfileEngineFile(string enginePath, int[] batchSizes,
      TensorRTBuildOptions options, int deviceId = -1)
  {
    return TensorRTEngine.LoadMultiProfileEngineFile(enginePath, batchSizes,
        options.UseCudaGraphs != 0, options.UseSpinWait != 0, deviceId);
  }


  /// <summary>
  /// Load a pre-built engine file (.engine).
  /// </summary>
  public TensorRTEngine LoadEngineFile(string enginePath, int batchSize, int deviceId = -1)
  {
    return TensorRTEngine.LoadEngineFile(enginePath, batchSize, deviceId);
  }


  /// <summary>
  /// Load from either ONNX or engine file based on extension.
  /// </summary>
  public TensorRTEngine Load(string path, int batchSize, TensorRTBuildOptions? options = null,
                              int deviceId = -1, string cacheDir = null, bool forceRebuild = false)
  {
    return TensorRTEngine.Load(path, batchSize, options, deviceId, cacheDir, forceRebuild);
  }


  /// <summary>
  /// Dispose the TensorRT runtime.
  /// </summary>
  public void Dispose()
  {
    if (disposed)
    {
      return;
    }

    disposed = true;

    if (initialized)
    {
      int result = TensorRTNative.Shutdown();
      if (result != 0)
      {
        Console.Error.WriteLine($"TensorRT shutdown warning: {TensorRTNative.GetLastErrorString()}");
      }
      initialized = false;
    }
  }
}
