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
using System.IO;
using System.Net;

using HtmlAgilityPack;

#endregion

namespace Ceres.Chess.LC0.NNFiles
{
  /// <summary>
  /// Manages downloading of LC0 weights files from website source.
  /// </summary>
  public class NNWeightsFileLC0Downloader
  {
    public static bool LogDownloadsToConsole = false;

    /// <summary>
    /// 
    /// </summary>
    public readonly string BaseURL;

    /// <summary>
    /// 
    /// </summary>
    public readonly string TargetDir;

    /// <summary>
    /// Constructor for downloader (from specified origin URL, and targeting specified directory on file system).
    /// </summary>
    /// <param name="baseURL"></param>
    /// <param name="targetDir"></param>
    public NNWeightsFileLC0Downloader(string baseURL, string targetDir)
    {
      BaseURL = baseURL;
      TargetDir = targetDir;
    }


    /// <summary>
    /// Downloads the network with specified ID.
    /// </summary>
    /// <param name="networkID"></param>
    /// <returns></returns>
    public string Download(string networkID = null)
    {
      // Download the file from the URL into a temporary file.
      string tempFile = Path.GetTempFileName();
      WebClient wc = new WebClient() { Proxy = null };
      wc.DownloadFile(BaseURL, tempFile);

      // Load the file as an HTML snippet.
      HtmlDocument htmlSnippet = LoadHtmlSnippetFromFile(tempFile);

      // Delete the temporary file.
      File.Delete(tempFile);

      // Scan the table to find this network.
      //< table class="table table-striped table-sm">
      //string tableName = "table table-striped table-sm"; //<div class="table-responsive">

      HtmlNode table = htmlSnippet.DocumentNode.SelectNodes("//table")[0];

      // 811 [13][5c8d14d5][2860.33][76747][6][64][2018-03-17 02:11:29.480216 -0400 EDT]
      //http://lczero.org/get_network?sha=a4b38d699d489003415bc58c52e1a5a600baa41af28ce798cc506a56ced9f51e

      // ALTERNATELY
      // http://lczero.org/networks --> http://lczero.org/get_network?sha=e040d0d3bc6ec0d4672ec569644b71a3a9209004d692833d9fcb4fd45737ed0c

      string fn = null;
      int downloadCount = 0;
      int i = 1;
      while (i < table.ChildNodes[3].ChildNodes.Count)
      {
        foreach (var ss in table.ChildNodes[3].ChildNodes[i].ChildNodes)
        {
          string link = null;
          if (ss.InnerHtml.Contains("get_network"))
          {
            link = ss.InnerHtml;
            //a href="/get_network?sha=6662b1ba3a8cc9d7938f8834344bb41f3a8d719061998cc93215c1e2036aa089" download="weights_418.txt.gz">6662b1ba</a>
            string sha = GetStr(link, link.IndexOf("sha=") + 4, link.IndexOf("download") - 2);
            string baseFN = TargetDir + "\\" + GetStr(link, link.IndexOf("download=") + 10, link.IndexOf("\">") - 3);
            fn = baseFN + ".gz";
            if (!new FileInfo(fn).Exists && (networkID == null || fn.Contains(networkID)))
            {
              string url = "http://training.lczero.org/get_network?sha=" + sha;
              if (LogDownloadsToConsole) Console.WriteLine("NNWeightsFileLC0Downloader downloading " + url + " --> " + fn);
              downloadCount++;
              wc.DownloadFile(url, fn);

              // Code if unpack was desired
              //Process unpack = Process.Start(@"gzip.exe", " -d " + fn);
              //unpack.WaitForExit();
              return fn;
            }
          }
        }
        i += 2;
      }

      return null;
    }

    #region Internal helpers

    static string GetStr(string str, int startIndex, int lastIndex)
    {
      return str.Substring(startIndex, lastIndex - startIndex);
    }

    /// <summary>
    /// Extract all anchor tags using HtmlAgilityPack
    /// </summary>
    /// <param name="htmlSnippet"></param>
    /// <returns></returns>
    private static List<string> ExtractAllAHrefTags(HtmlDocument htmlSnippet)
    {
      List<string> hrefTags = new List<string>();

      foreach (HtmlNode link in htmlSnippet.DocumentNode.SelectNodes("//a[@href]"))
      {
        HtmlAttribute att = link.Attributes["href"];
        hrefTags.Add(att.Value);
      }

      return hrefTags;
    }

    private static HtmlDocument LoadHtmlSnippetFromFile(string fn)
    {
      TextReader reader = System.IO.File.OpenText(fn);

      HtmlDocument doc = new HtmlDocument();
      doc.Load(reader);

      reader.Close();

      return doc;
    }

    #endregion
  }


}


#if SAMPLE_TABLE_ENTRY
      <thead>
      <tr>
        <th>Id</th>
        <th>Network</th>
        <th>Elo</th>
        <th>Games</th>
        <th>Blocks</th>
        <th>Filters</th>
        <th>Time</th>
      </tr>
    </thead>
    <tbody>
      
      <tr>
        <td>418</td>
        <td><a href="/get_network?sha=6662b1ba3a8cc9d7938f8834344bb41f3a8d719061998cc93215c1e2036aa089" download="weights_418.txt.gz">6662b1ba</a></td>
        <td>5834.61</td>
        <td>31147</td>
        <td>15</td>
        <td>195</td>
        <td>2018-06-17 10:17:12.963156 -0400 EDT</td>
      </tr>
#endif
