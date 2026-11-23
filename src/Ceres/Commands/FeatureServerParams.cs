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
  /// Usage: Ceres SERVER [net=NONE] [device=GPU:0] [port=50055] [maxclients=64] [instances=1] [maxevaluators=64]
  ///
  /// All parameters are optional. Clients always send their own network and device specification
  /// in the handshake, so NET/DEVICE are only fallbacks for clients that omit them
  /// (NET=NONE: no default network).
  ///
  /// MAXCLIENTS is the maximum number of concurrent connections. Each MCGS engine opens two
  /// connections per server (dual evaluators), so a tournament game stream with two players
  /// evaluating on this server uses four connections.
  ///
  /// INSTANCES is the number of instances built of each evaluator (default 1: the batches of all clients
  /// of an evaluator execute one at a time). With INSTANCES=2 the two connections of an MCGS engine each
  /// get their own instance, so their two batches in flight execute concurrently on the device(s).
  /// Whether that is faster depends on the device and on the other servers the client uses
  /// (see NNRemoteServer.InstancesPerEvaluator); it needs a second set of execution contexts in device memory.
  ///
  /// MAXEVALUATORS is the maximum number of evaluators held at once. One is needed per distinct
  /// network on each device (or bundle of devices), for each concurrent stream placed there
  /// (see NNDeviceSlots; each such evaluator may in turn have up to INSTANCES instances).
  /// </summary>
  public static class FeatureServerParams
  {
    public static void ParseAndExecute(string args)
    {
      KeyValueSetParsed keys = new KeyValueSetParsed(args,
        new System.Collections.Generic.List<string>
        {
          "NET", "DEVICE", "PORT", "MAXCLIENTS", "INSTANCES", "MAXEVALUATORS"
        });

      string netSpec = keys.GetValueOrDefault("NET", "NONE", false);
      string deviceSpec = keys.GetValueOrDefault("DEVICE", "GPU:0", false);
      string portStr = keys.GetValueOrDefault("PORT", NNRemoteProtocol.DEFAULT_PORT.ToString(), false);
      string maxClientsStr = keys.GetValueOrDefault("MAXCLIENTS", NNRemoteServer.DEFAULT_MAX_CLIENTS.ToString(), false);
      string instancesStr = keys.GetValueOrDefault("INSTANCES", "1", false);
      string maxEvaluatorsStr = keys.GetValueOrDefault("MAXEVALUATORS", NNRemoteServer.DEFAULT_MAX_CACHED_EVALUATORS.ToString(), false);

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

      if (!int.TryParse(instancesStr, out int instances) || instances < 1)
      {
        DispatchCommands.ShowErrorExit($"Invalid INSTANCES value: {instancesStr}");
        return;
      }

      if (!int.TryParse(maxEvaluatorsStr, out int maxEvaluators) || maxEvaluators < 1)
      {
        DispatchCommands.ShowErrorExit($"Invalid MAXEVALUATORS value: {maxEvaluatorsStr}");
        return;
      }

      // NET=NONE means no default network (clients must specify their own).
      string defaultNet = netSpec.Equals("NONE", StringComparison.OrdinalIgnoreCase) ? null : netSpec;

      NNRemoteServer server = new NNRemoteServer(
        port: port,
        maxClients: maxClients,
        defaultNetworkSpec: defaultNet,
        defaultDeviceSpec: deviceSpec,
        instancesPerEvaluator: instances,
        maxCachedEvaluators: maxEvaluators);

      server.Start(); // Blocks until Ctrl+C.
    }
  }
}
