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

using System.IO;

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Static helper methods relating to files and directories.
  /// </summary>
  public static class FileUtils
  {
    /// <summary>
    /// Returns if all of the semicolon separated paths exist.
    /// </summary>
    /// <param name="paths"></param>
    /// <returns></returns>
    public static bool PathsListAllExist(string paths)
    {
      if (paths == null)
      {
        return true;
      }

      // Verify each of the parths exists
      string[] parts = paths.Split(new char[] { ';', ',' });
      foreach (string part in parts)
      {
        if (!Directory.Exists(part))
        {
          return false;
        }
      }

      return true;
    }
  }
}
