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

#endregion

namespace Ceres.Chess.Data.Nets
{
  public enum ReferenceNetType
  {
    LC0,
    Ceres
  }

  /// <summary>
  /// Describes a reference network.
  /// </summary>
  public readonly record struct ReferenceNetInfo
  {
    /// <summary>
    /// Short descriptive net ID.
    /// </summary>
    public readonly string ID;

    /// <summary>
    /// Type of net (e.g. LC0 or Ceres).
    /// </summary>
    public readonly ReferenceNetType NetType;

    /// <summary>
    /// Specification string used to load the net using NNEvaluator.FromSpecification.
    /// </summary>
    public readonly string NetSpecificationString { get; }

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="netType"></param>
    /// <param name="netSpecificationString"></param>
    public ReferenceNetInfo(string id, ReferenceNetType netType, string netSpecificationString)
    {
      ID = id;
      NetType = netType;
      NetSpecificationString = netSpecificationString;
    } 
  }
}
