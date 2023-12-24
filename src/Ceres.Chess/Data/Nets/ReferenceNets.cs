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

using System.IO;
using System.Collections.Generic;

using Ceres.Chess.UserSettings;
using Ceres.Chess.LC0.NNFiles;

#endregion

namespace Ceres.Chess.Data.Nets
{
  public static class ReferenceNets
  { 
    /// <summary>
    /// Set of commonly used reference nets (baselines at various stages of LC0 development).
    /// </summary>
    public static Dictionary<string, ReferenceNetInfo> Baselines = new()
      {
//        { "32930", SimpleLC0Net("32930") },
        { "42850", SimpleLC0Net("42850") },
        { "703810", SimpleLC0Net("703810") },
        { "753723", SimpleLC0Net("753723") },
        { "606512", SimpleLC0Net("606512") },
        { "784984", SimpleLC0Net("784984") },
        { "809942", SimpleLC0Net("809942") }, // 801307
        { "811971", SimpleLC0Net("811971") }, //Training restarted after surgery after 811971

        {"BT2", ONNXNet32LC0("BT2-768x15smolgen-12h-do-01-swa-onnx-2350000-rule50.gz")},
        {"T1_DISTILL_256", ONNXNet16LC0("t1-256x10-distilled-swa-2432500_fp16")},
        {"T1_DISTILL_512_10_FP32", ONNXNet32LC0("t1-512x15x8h-distilled-swa-3395000")},
        {"T1_DISTILL_512_10_FP16", ONNXNet16LC0("t1-512x15x8h-distilled-swa-3395000_fp16")},

        {"T1_768_PF16", ONNXNet16LC0("t1-768x15x24h-swa-4000000_fp16")},
        {"T2", ONNXNet16LC0("t2-768x15x24h-swa-5230000.pb.gz_fp16")},
        {"T3", ONNXNet32LC0("t3-512x15x16h-swa-2815000") }
      //        {"BT4", ONNXNet32LC0("BT4-1024x15x32h-swa-365000#32")},
    };


    /// <summary>
    /// Checks if a net is downloaded, and downloads it if not.
    /// </summary>
    /// <param name="netName"></param>
    public static void CheckDownloaded(string netName)
    {
      NNWeightsFileLC0.LookupOrDownload(netName);
    }


    #region Internal helpers

    static ReferenceNetInfo SimpleLC0Net(string netID) => new ReferenceNetInfo(netID, ReferenceNetType.LC0, netID);
    static ReferenceNetInfo ONNXNet16LC0(string netID) =>
      new ReferenceNetInfo(netID,
                           ReferenceNetType.LC0,
                           @"ONNX_ORT:" + Path.Combine(CeresUserSettingsManager.Settings.DirLC0Networks, netID) + "#16");
    static ReferenceNetInfo ONNXNet32LC0(string netID) =>
      new ReferenceNetInfo(netID,
                           ReferenceNetType.LC0,
                           @"ONNX_ORT:" + Path.Combine(CeresUserSettingsManager.Settings.DirLC0Networks, netID) + "#32");

    #endregion
  }
}
