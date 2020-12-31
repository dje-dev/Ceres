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

namespace Ceres.Base.Environment
{
  /// <summary>
  /// Extracts metadata relating to Git version control.
  /// </summary>
  public static class GitInfo
  {
    /// <summary>
    /// Returns a short description version of the current Git version of this code.
    /// </summary>
    public static string VersionString
    {
      get
      {
        string gitString;
        if (ThisAssembly.Git.Tag != "")
          gitString = $"git Tag:{ThisAssembly.Git.Tag}";
        else
          gitString = $"git Branch:{ThisAssembly.Git.Branch} Commit:{ThisAssembly.Git.Commit} {(ThisAssembly.Git.IsDirty ? "(dirty)" : "")}";
        return gitString;
      }
    }
  }
}


