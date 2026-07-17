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

#endregion

namespace Ceres.MCGS.Search.QProbeSelect;

/// <summary>
/// Lock-guarded static store of loaded QProbeSelectNN models, keyed by file path
/// (mirrors the FPURunningStats lazy-shared-evaluator pattern). Each model is
/// loaded and validated once per process and shared by all engines; a header
/// line is logged on first load so runs are self-documenting.
/// </summary>
public static class QProbeSelectModelStore
{
  static readonly object lockObj = new object();
  static readonly Dictionary<string, QProbeSelectNNModel> loadedModels = new Dictionary<string, QProbeSelectNNModel>();


  /// <summary>
  /// Returns the model for the given path, loading (and logging the header) on first use.
  /// </summary>
  public static QProbeSelectNNModel EnsureLoaded(string path)
  {
    string fullPath = Path.GetFullPath(path);
    lock (lockObj)
    {
      if (loadedModels.TryGetValue(fullPath, out QProbeSelectNNModel existing))
      {
        return existing;
      }

      QProbeSelectNNModel model = QProbeSelectNNModel.Load(fullPath);
      loadedModels[fullPath] = model;

      Console.WriteLine($"QProbeSelectNN loaded: file={Path.GetFileName(fullPath)} "
                      + $"params={model.NumParams:N0} act={model.Activation} "
                      + $"schemaSha1={Convert.ToHexString(model.SchemaSHA1).ToLowerInvariant()} "
                      + $"nnNetHash={model.NNNetHash:x}");
      return model;
    }
  }
}
