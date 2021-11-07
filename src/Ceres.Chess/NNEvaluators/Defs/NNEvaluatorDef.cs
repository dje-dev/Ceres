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
using System.Linq;
using Ceres.Chess;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.LC0;
using Chess.Ceres.NNEvaluators;
using Ceres.Chess.NNEvaluators.Specifications;

#endregion

namespace Ceres.Chess.NNEvaluators.Defs
{
  /// <summary>
  /// Defines all of the characteristics of a neural network evaluator which 
  /// such as the underlying network(s) and the devices on which the evaluation will run.
  /// 
  /// This definition can be used to create an NNEvaluator 
  /// (for example, byusing NNEvaluatorBuilder).
  /// </summary>
  [Serializable]
  public class NNEvaluatorDef
  {
    /// <summary>
    /// Default minimum batch size before a batch can be 
    /// possibly split over multiple evaluators.
    /// </summary>
    const int DEFAULT_MIN_SPLIT_POSITIONS = 32;

    /// <summary>
    /// Array of devices used for the evaluation.
    /// </summary>
    public readonly (NNEvaluatorDeviceDef Device, float Fraction)[] Devices;

    /// <summary>
    /// Method used to combine (possibly) multiple devices.
    /// </summary>
    public NNEvaluatorDeviceComboType DeviceCombo = NNEvaluatorDeviceComboType.Single;

    /// <summary>
    /// Array of networks to be used for the evaluation, with corresponding
    /// weights of their value, policy and MLH heads in the overall evaluator.
    /// </summary>
    public readonly (NNEvaluatorNetDef Net, float WeightValue, float WeightPolicy, float WeightM)[] Nets;

    /// <summary>
    /// Method used to combine (possibly) multiple nets.
    /// </summary>
    public readonly NNEvaluatorNetComboType NetCombo = NNEvaluatorNetComboType.Single;

    /// <summary>
    /// Minimum number of positions before a batch is possibly split over multiple devices.
    /// </summary>
    public int MinSplitNumPositions = DEFAULT_MIN_SPLIT_POSITIONS;

    /// <summary>
    /// If certain the supplemental (internal) layers should also be retrieved.
    /// </summary>
    public bool RetrieveSupplementalLayers = false;

    /// <summary>
    /// Caching mode (if evaluations are saved to memory and/or disk).
    /// </summary>
    public PositionEvalCache.CacheMode CacheMode = PositionEvalCache.CacheMode.None;

    // TODO: move this to NNEvaluatorSet
    public PositionEvalCache PreloadedCache;

    /// <summary>
    /// If not null then all evaluators built from definitions
    /// having this same name are shared.
    /// </summary>
    public readonly string SharedName;

    /// <summary>
    /// Returns if the evaluator is shared (under a specified name).
    /// </summary>
    public bool IsShared => SharedName != null;


    public enum PositionTransformType { None, Mirror };

    /// <summary>
    /// Type of transformation (if any) to apply to position before evaluation.
    /// </summary>
    public PositionTransformType PositionTransform = PositionTransformType.None;

    public enum LocationType { Local, Remote };

    /// <summary>
    /// Type of evaluator to use (local or remot).
    /// </summary>
    public LocationType Location = LocationType.Local;

    //[NonSerialized]
    //public Func<SearchContext, int> DynamicNNSelectorFunc;

    public int NumDevices => Devices.Length;

    public string CacheFileName = "Ceres.cache.dat";

    /// <summary>
    /// Returns array of the indicies of the devices to be used for the evaluator.
    /// </summary>
    public int[] DeviceIndices
    {
      get
      {
        int[] ret = new int[Devices.Length];
        for (int i = 0; i < Devices.Length; i++)
        {
          ret[i] = Devices[i].Device.DeviceIndex;
        }
        return ret;
      }
    }
    
    // ..........................................................
    /// <summary>
    /// When hashing in cache, how many positions should be included in the hash.
    /// History planes are included as this value goes over 1.
    /// 
    /// Higher numbers produce more precise results (less false cache matches)
    /// but reduce the number of caching opportunities and therefore require more NN evaluations.
    /// 
    /// Testing shows that choosing 1 rather than 2 only very slightly decreases 
    /// play performance at fixed nodes, but significantly decreases overall runtime (by 10% to 20%).
    /// 
    /// Note that our hash function already includes en passant flags taken from the current position.
    /// </summary>
    public int NumCacheHashPositions = 1;

    /// <summary>
    /// The mode to be used for determining if the 50 move rule is 
    /// incorporated into the hash function (and therefore affects cache equality tests).
    /// </summary>
    public PositionMiscInfo.HashMove50Mode HashMode = PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="devices"></param>
    /// <param name="deviceCombo"></param>
    /// <param name="netCombo"></param>
    /// <param name="sharedName"></param>
    /// <param name="nets"></param>
    public NNEvaluatorDef(IEnumerable<(NNEvaluatorDeviceDef, float)> devices, 
                          NNEvaluatorDeviceComboType deviceCombo, NNEvaluatorNetComboType netCombo,
                          string sharedName, 
                          params (NNEvaluatorNetDef net, float weightValue, float weightPolicy, float weightM)[] nets)
    {
      Devices = devices.ToArray();
      Nets = nets;
      DeviceCombo = deviceCombo;
      NetCombo = netCombo;
      SharedName = sharedName;
    }


    /// <summary>
    /// Constructor from single specified net (by network) and single device.
    /// </summary>
    /// <param name="netType"></param>
    /// <param name="networkID"></param>
    /// <param name="deviceType"></param>
    /// <param name="deviceIndex"></param>
    /// <param name="sharedName"></param>
    public NNEvaluatorDef(NNEvaluatorType netType, string networkID,
                          NNDeviceType deviceType = NNDeviceType.GPU, int deviceIndex = 0, string sharedName = null)
    {
      if (networkID == null)
      {
        throw new ArgumentNullException(nameof(networkID));
      }

      Devices = new (NNEvaluatorDeviceDef, float)[] { (new NNEvaluatorDeviceDef(deviceType, deviceIndex), 1.0f) };
      Nets = new (NNEvaluatorNetDef net, float weightValue, float weightPolicy, float weightM)[] { (new NNEvaluatorNetDef(networkID, netType, NNEvaluatorPrecision.FP16), 1.0f, 1.0f, 1.0f) };
      DeviceCombo = NNEvaluatorDeviceComboType.Single;
      SharedName = sharedName;
    }



    /// <summary>
    /// Constructor from specified single net and device combo and devices.
    /// </summary>
    /// <param name="net"></param>
    /// <param name="deviceCombo"></param>
    /// <param name="sharedName"></param>
    /// <param name="devices"></param>
    public NNEvaluatorDef(NNEvaluatorNetDef net, NNEvaluatorDeviceComboType deviceCombo, string sharedName, params NNEvaluatorDeviceDef[] devices)
    {
      Devices = new (NNEvaluatorDeviceDef net, float fraction)[devices.Length];
      for (int i = 0; i < devices.Length; i++)
      {
        Devices[i] = new(devices[i], 1.0f / devices.Length);
      }

      Nets = new (NNEvaluatorNetDef net, float weightValue, float weightPolicy, float weightM)[] { (net, 1.0f, 1.0f, 1.0f) };
      DeviceCombo = deviceCombo;
      SharedName = sharedName;
    }


    public NNEvaluatorDef(NNEvaluatorNetDef net, string sharedName = null, params (NNEvaluatorDeviceDef deviceDef, float fraction)[] devices)
    {
      Devices = devices.ToArray();
      Nets = new (NNEvaluatorNetDef net, float weightValue, float weightPolicy, float weightM)[] { (net, 1.0f, 1.0f, 1.0f) };
      DeviceCombo = devices.Length == 0 ? NNEvaluatorDeviceComboType.Single : NNEvaluatorDeviceComboType.Pooled;
      SharedName = sharedName;
    }


    public NNEvaluatorDef(NNEvaluatorNetDef net, NNEvaluatorDeviceComboType deviceCombo, string sharedName, params (NNEvaluatorDeviceDef deviceDef, float fraction)[] devices)
    {
      Devices = devices.ToArray();
      DeviceCombo = deviceCombo;

      Nets = new (NNEvaluatorNetDef net, float weightValue, float weightPolicy, float weightM)[] { (net, 1.0f, 1.0f, 1.0f) };
      NetCombo = NNEvaluatorNetComboType.Single;
      SharedName = sharedName;
    }

    public NNEvaluatorDef(NNEvaluatorNetComboType netCombo, NNEvaluatorDeviceDef device, string sharedName, params (NNEvaluatorNetDef netDef, float weightValue, float weightPolicy, float weightM)[] netDefs)
    {
      NetCombo = netCombo;
      Nets = netDefs;

      DeviceCombo = NNEvaluatorDeviceComboType.Single;
      Devices = new (NNEvaluatorDeviceDef device, float fraction)[] { (device, 1) };
      SharedName = sharedName;
    }


    public NNEvaluatorDef(NNEvaluatorNetComboType netCombo, IEnumerable<(NNEvaluatorNetDef, float, float, float)> nets, 
                          NNEvaluatorDeviceComboType deviceCombo, IEnumerable<(NNEvaluatorDeviceDef, float)> devices,
                          string sharedName)
    {
      Devices = devices.ToArray();
      DeviceCombo = deviceCombo;

      Nets = nets.ToArray();
      NetCombo = netCombo;
      SharedName = sharedName;
    }

    #region Static factory methods

    /// <summary>
    /// Returns an NNEvaluatorDef corresponding to speciifed strings with network and device specifications.
    /// </summary>
    /// <param name="netSpecificationString"></param>
    /// <param name="deviceSpecificationString"></param>
    /// <returns></returns>
    public static NNEvaluatorDef FromSpecification(string netSpecificationString, string deviceSpecificationString)
    {
      return NNEvaluatorDefFactory.FromSpecification(netSpecificationString, deviceSpecificationString);
    }


    /// <summary>
    /// Creates and returns an evaluator based on this definition.
    /// </summary>
    /// <returns></returns>
    public NNEvaluator ToEvaluator() => NNEvaluatorFactory.BuildEvaluator(this);
    

    #endregion

    #region Equality/equivalence testing 

    /// <summary>
    /// Returns if this evaluator generates identical network evaluations to another NNEvaluatorDef.
    /// </summary>
    /// <param name="otherDef"></param>
    /// <returns></returns>
    public bool NetEvaluationsIdentical(NNEvaluatorDef otherDef)
    {
      if (NetCombo != otherDef.NetCombo) return false;
      if (Nets.Length != otherDef.Nets.Length) return false;
      if (PositionTransform != otherDef.PositionTransform) return false;

      for (int i = 0; i < Nets.Length; i++)
      {
        if (Nets[i] != otherDef.Nets[i])
          return false;
      }
      return true;
    }

    /// <summary>
    /// Returns if the fractional allocations to the devices are uniform.
    /// </summary>
    public bool EqualFractions
    {
      get
      {
        bool equalFractions = true;
        for (int i = 1; i < Devices.Length; i++)
        {
          if (Devices[i].Fraction != Devices[0].Fraction)
            equalFractions = false;
        }
        return equalFractions;
      }
    }

    #endregion

    /// <summary>
    /// Modifies the evaluator definition to point to a specified network
    /// instead of the network currently specified.
    /// </summary>
    /// <param name="networkID"></param>
    public void TryModifyNetworkID(string networkID)
    {
      if (Nets.Length > 1)
      {
        throw new Exception("TryModifyNetworkID only supported with single nets");
      }

      Nets[0].Net = Nets[0].Net with { NetworkID = networkID };
    }

    /// <summary>
    /// Modifies the evaluator definition to point to a specified device
    /// instead of the device currently specified.
    /// </summary>
    /// <param name="deviceID"></param>
    public void TryModifyDeviceID(int deviceID)
    {
      if (DeviceCombo == NNEvaluatorDeviceComboType.Pooled)
      {
        // Nothing to do, we will be sharing the same evaluator
        return;
      }

      if (DeviceCombo != NNEvaluatorDeviceComboType.Single)
      {
        throw new Exception($"Device combo must be single, was: {DeviceCombo}");
      }

      if (Devices.Length > 1)
      {
        throw new Exception($"Must be single device, was cardinality: {Devices.Length}");
      }

      Devices[0].Device.DeviceIndex = deviceID;
    }


    /// <summary>
    /// Returns string summary of evaluator definition.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string ret = $"<NNEvaluatorDef {NNNetSpecificationString.ToSpecificationString(NetCombo, Nets)} " +
                   $"{NNDevicesSpecificationString.ToSpecificationString(DeviceCombo, Devices) } ";
      if (PositionTransform != PositionTransformType.None)
      {
        ret += PositionTransform.ToString() + " ";
      }

      if (IsShared)
      {
        ret += "=" + SharedName;
      }

      return ret + ">";
    }


  }
}
