#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.Base.Environment
{
  /// <summary>
  /// Interface for an application event logger.
  /// </summary>
  public interface IAppLogger
  {
    /// <summary>
    /// Logs an informational event.
    /// </summary>
    /// <param name="category"></param>
    /// <param name="message"></param>
    /// <param name="instanceID"></param>
    void LogInfo(string category, string message, int instanceID = -1);

    /// <summary>
    /// Logs a warning event.
    /// </summary>
    /// <param name="category"></param>
    /// <param name="message"></param>
    /// <param name="instanceID"></param>
    void LogWarn(string category, string message, int instanceID = -1);

    /// <summary>
    /// Logs an error event.
    /// </summary>
    /// <param name="category"></param>
    /// <param name="message"></param>
    /// <param name="instanceID"></param>
    void LogError(string category, string message, int instanceID = -1);
  }
}
