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

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Static helper methods for working with strings.
  /// </summary>
  public static class StringUtils
  {
    /// <summary>
    /// Compiled regular expression that squashes repeated characters.
    /// </summary>
    static readonly Regex repeatedCharTrimmer = new Regex(@"\s\s+", RegexOptions.Compiled);

    /// <summary>
    /// Returns the specified string with all extraneous spaces removed
    /// (repeated spaces in the middle of the string are compacted).
    /// </summary>
    /// <param name="str"></param>
    /// <returns></returns>
    public static string WhitespaceRemoved(string str) => repeatedCharTrimmer.Replace(str, " ");


    /// <summary>
    /// Returns a string truncated if necessary to fit within specified length.
    /// </summary>
    /// <param name="s"></param>
    /// <param name="len"></param>
    /// <returns></returns>
    public static string TrimmedIfNeeded(string s, int len) => s.Length > len ? s.Substring(0, len) : s;


    /// <summary>
    /// Adjust a string's length to be exactly specified value,
    /// truncating or padding on the right as needed.
    /// </summary>
    /// <param name="s"></param>
    /// <param name="len"></param>
    /// <returns></returns>
    public static string Sized(string s, int len)
    {
      if (s.Length > len) return s.Substring(0, len);

      while (s.Length < len) s = s + " ";
      return s;
    }


    /// <summary>
    /// Launches browser on a specified URL in a platform-appropriate way.
    /// 
    /// With thanks to Brock Allen
    /// (https://brockallen.com/2016/09/24/process-start-for-urls-on-net-core/).
    /// </summary>
    /// <param name="url"></param>
    public static void LaunchBrowserWithURL(string url)
    {
      // hack because of this: https://github.com/dotnet/corefx/issues/10361
      if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
      {
        url = url.Replace("&", "^&");
        Process.Start(new ProcessStartInfo("cmd", $"/c start {url}") { CreateNoWindow = true });
      }
      else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
      {
        Process.Start("xdg-open", url);
      }
      else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
      {
        Process.Start("open", url);
      }
      else
      {
        // Probably wont' work....
        Process.Start(url);
      }
    }

  }
}
