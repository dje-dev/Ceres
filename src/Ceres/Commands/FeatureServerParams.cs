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

using Ceres.Base.DataTypes;
using Ceres.Chess.NNEvaluators.Remote;

#endregion

namespace Ceres.Commands
{
  /// <summary>
  /// Parses and executes the SERVER command to start a remote NN evaluation server.
  ///
  /// Usage: Ceres SERVER net=LC0:42767 device=GPU:0,1 [port=50055] [maxclients=4]
  /// </summary>
  public static class FeatureServerParams
  {
    public static void ParseAndExecute(string args)
    {
      KeyValueSetParsed keys = new KeyValueSetParsed(args,
        new System.Collections.Generic.List<string>
        {
          "NET", "DEVICE", "PORT", "MAXCLIENTS"
        });

      string netSpec = keys.GetValue("NET");
      string deviceSpec = keys.GetValue("DEVICE");
      string portStr = keys.GetValueOrDefault("PORT", NNRemoteProtocol.DEFAULT_PORT.ToString(), false);
      string maxClientsStr = keys.GetValueOrDefault("MAXCLIENTS", "4", false);

      if (netSpec == null)
      {
        DispatchCommands.ShowErrorExit("SERVER command requires NET parameter, e.g. NET=LC0:42767");
        return;
      }

      if (deviceSpec == null)
      {
        DispatchCommands.ShowErrorExit("SERVER command requires DEVICE parameter, e.g. DEVICE=GPU:0,1");
        return;
      }

      if (!int.TryParse(portStr, out int port))
      {
        DispatchCommands.ShowErrorExit($"Invalid PORT value: {portStr}");
        return;
      }

      if (!int.TryParse(maxClientsStr, out int maxClients))
      {
        DispatchCommands.ShowErrorExit($"Invalid MAXCLIENTS value: {maxClientsStr}");
        return;
      }

      // NET=NONE means no default network (clients must specify their own).
      string defaultNet = netSpec.Equals("NONE", StringComparison.OrdinalIgnoreCase) ? null : netSpec;

      NNRemoteServer server = new NNRemoteServer(
        port: port,
        maxClients: maxClients,
        defaultNetworkSpec: defaultNet,
        defaultDeviceSpec: deviceSpec);

      server.Start(); // Blocks until Ctrl+C.
    }
  }
}
