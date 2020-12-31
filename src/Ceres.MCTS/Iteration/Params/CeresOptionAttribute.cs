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

namespace Ceres.MCTS.Params
{
  /// <summary>
  /// Attribute applied to options fields that adjust the runtime behavior of Ceres.
  /// </summary>
  public sealed class CeresOptionAttribute : Attribute
  {
    /// <summary>
    /// Underlying data type
    /// </summary>
    public Type Type;

    /// <summary>
    /// Category into which this option follows
    /// </summary>
    public string Category;

    /// <summary>
    /// Short name of the attribute (used in command line settings)
    /// </summary>
    public string Name;

    /// <summary>
    /// Descriptive text
    /// </summary>
    public string Desc;

    /// <summary>
    /// Default value of the option (expressed as a string)
    /// </summary>
    public string Default;

    /// <summary>
    /// If the option is hidden by default
    /// </summary>
    public bool Hidden = false;

    /// <summary>
    /// If the option must be specified on the command line
    /// </summary>
    public bool Required = false;

    // --------------------------------------------------------------------------------------------
    public override string ToString()
    {
      return $"<CeresOptionAttribute {Category}-{Name} {Desc} Default={Default}";
    }
  }
}
