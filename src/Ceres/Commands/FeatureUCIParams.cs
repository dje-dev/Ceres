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

    public static FeatureUCIParams ParseUCICommand(string args)
    {
      KeyValueSetParsed keys = new KeyValueSetParsed(args, new string[] { "NETWORK", "DEVICE", "PRUNING" });

      return new FeatureUCIParams()
      {
        NetworkSpec = keys.GetValueOrDefaultMapped<NNNetSpecificationString>("Network", CeresUserSettingsManager.Settings.DefaultNetworkSpecString, true, spec => new NNNetSpecificationString(spec)),
        DeviceSpec = keys.GetValueOrDefaultMapped("Device", CeresUserSettingsManager.Settings.DefaultDeviceSpecString, true, spec => new NNDevicesSpecificationString(spec)),
        Pruning = keys.GetValueOrDefaultMapped<bool>("Pruning", "true", false, str => bool.Parse(str))
      };
    }

  }
}
