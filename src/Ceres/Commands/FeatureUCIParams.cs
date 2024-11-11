#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using Ceres.Base.DataTypes;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.Commands
{
  public record FeatureUCIParams
  {
    public NNNetSpecificationString NetworkSpec { init; get; }
    public NNDevicesSpecificationString DeviceSpec { init; get; }
    public bool Pruning { init; get; }

    
    public FeatureUCIParams(NNNetSpecificationString netSpec, NNDevicesSpecificationString devicesSpec, bool pruning)
    {
      NetworkSpec = netSpec;
      DeviceSpec = devicesSpec;
      Pruning = pruning;
    }


    public static FeatureUCIParams ParseUCICommand(string args)
    {
      KeyValueSetParsed keys = new KeyValueSetParsed(args, new string[] { "NETWORK", "DEVICE", "PRUNING" });

      return new FeatureUCIParams(
        keys.GetValueOrDefaultMapped("Network", null, false, spec => new NNNetSpecificationString(spec)),
        keys.GetValueOrDefaultMapped("Device", "GPU:0", false, spec => new NNDevicesSpecificationString(spec)),
        keys.GetValueOrDefaultMapped("Pruning", "true", false, str => bool.Parse(str)));      
    }

  }
}
