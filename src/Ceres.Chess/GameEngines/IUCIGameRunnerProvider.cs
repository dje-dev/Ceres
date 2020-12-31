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

using Ceres.Chess.ExternalPrograms.UCI;

#endregion

namespace Ceres.Chess.GameEngines
{
  /// <summary>
  /// Interface implemented by objects that can supply an associated UCIRunner.
  /// </summary>
  public interface IUCIGameRunnerProvider
  {
    UCIGameRunner UCIRunner { get; }
  }
}
