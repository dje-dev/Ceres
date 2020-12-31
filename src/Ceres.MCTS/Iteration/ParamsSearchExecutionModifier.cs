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
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Represents an Action which can act upon a ParamsSearchExecution,
  /// uniquely identified by a string identifier.
  /// 
  /// This design allows safe serialization of delegates (within process)
  /// because they are identified by strings, which serialize directly.
  /// 
  /// TODO: this could be generalized to be a generic class.
  /// </summary>
  [Serializable]
  public class ParamsSearchExecutionModifier
  {
    [NonSerialized]
    static Dictionary<string, Action<ParamsSearchExecution>> DictModifiers = new();

    /// <summary>
    /// Unique identifier.
    /// </summary>
    public readonly string ID;

    /// <summary>
    /// Global registeres a modifier.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="modifier"></param>
    public static void Register(string id, Action<ParamsSearchExecution> modifier)
    {
      if (DictModifiers.ContainsKey(id))
        throw new Exception($"Already registered with ParamsSearchExecutionModifier.Register: {id}");

      DictModifiers[id] = modifier;
    }

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    public ParamsSearchExecutionModifier(string id)
    {
      if (!DictModifiers.ContainsKey(id)) throw new Exception("Modifier must be already registered with ParamsSearchExecutionModifier.Register");

      ID = id;
    }

    /// <summary>
    /// Invokes the modifier on specified object.
    /// </summary>
    /// <param name="parms"></param>
    public void Invoke(ParamsSearchExecution parms)
    {
      if (!DictModifiers.TryGetValue(ID, out Action<ParamsSearchExecution> modifier))
        throw new Exception("Modifier no registered with ParamsSearchExecutionModifier.Register");

      modifier(parms);
    }
  }
}


