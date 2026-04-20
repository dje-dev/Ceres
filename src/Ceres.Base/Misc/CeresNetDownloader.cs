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
using System.IO;
using Spectre.Console;
using System.Net.Http;
using System.Threading.Tasks;
using System.IO.Compression;
using Ceres.Base.Benchmarking;
using Ceres.Base.Misc;


#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Manages downloading of CeresNets from the CeresNets GitHub repository.
  /// </summary>
  public class CeresNetDownloader
  {
    /// <summary>
    /// Fixed URL for downloading CeresNets from the CeresNets GitHub repository.
    /// </summary>
    const string CERES_SOURCE_URL = @"https://github.com/dje-dev/CeresNets/releases/download/";

    /// <summary>
    /// If user requests cancellation with Ctrl-C.
    /// </summary>
    bool shouldCancel = false;

    /// <summary>
    /// HttpClient used for downloading CeresNets.
    /// </summary>
    HttpClient client = new HttpClient();


    /// <summary>
    /// Test method to download one of the small Ceres nets.
    /// </summary>
    public static void Test()
    {
      string CERES_NET_DIR = @"e:\cout\pubnets";
      string CERES_NET_ID = "C1-256-10";

      using (new TimingBlock("download"))
      {
        CeresNetDownloader downloader = new CeresNetDownloader();
        downloader.DownloadCeresNetIfNeeded(CERES_NET_ID, CERES_NET_DIR, false);
        Console.WriteLine($"uci info auto-downloaded {CERES_NET_ID} from CeresNets repository into {CERES_NET_DIR}");

      }

      System.Environment.Exit(3);
    }


    /// <summary>
    /// Downloads a CeresNet from the CeresNets GitHub repository 
    /// if it is not already present in the specified directory.
    /// </summary>
    /// <param name="ceresNetID"></param>
    /// <param name="ceresNetDirectory"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public (bool alreadyDownloaded, string fullNetworkPath) DownloadCeresNetIfNeeded(string ceresNetID, string ceresNetDirectory, bool uciMessagesOnly)
    {
      if (!Directory.Exists(ceresNetDirectory))
      {
        throw new Exception($"CeresNetDir does not exist ({ceresNetDirectory})");
      }

      string targetFileName = Path.Combine(ceresNetDirectory, ceresNetID);
      if (!targetFileName.ToUpper().EndsWith(".ONNX"))
      {
        targetFileName += ".onnx";
      }

      if (File.Exists(targetFileName))
      {
        return (true, targetFileName);
      }

      if (uciMessagesOnly)
      {
        Console.WriteLine($"uci info begin auto-download of {ceresNetID} from CeresNets repository into {ceresNetDirectory}");
      }

      return (false, DoDownloadNet(ceresNetID, ceresNetDirectory, ref targetFileName, uciMessagesOnly));
    }


    /// <summary>
    ///  Performs the download and unpacking of the CeresNet.
    /// </summary>
    /// <param name="ceresNetID"></param>
    /// <param name="ceresNetDirectory"></param>
    /// <param name="targetFileName"></param>
    /// <returns></returns>
    private string DoDownloadNet(string ceresNetID, string ceresNetDirectory, ref string targetFileName, bool uciMessagesOnly)
    {
      // Ceres nets always uppercase (GitHub download is Linux-based and case-sensitive).
      ceresNetID = ceresNetID.ToUpper();

      IAnsiConsole console = uciMessagesOnly ? null : AnsiConsole.Console;

      // Allow user to cancel download with Ctrl-C.
      Console.CancelKeyPress += (sender, e) =>
      {
        shouldCancel = true;
      };

      string tempDirectory = Directory.CreateTempSubdirectory().FullName;

      try
      {
        DownloadAndUnpackFile(ceresNetID, ceresNetDirectory, $"{CERES_SOURCE_URL}{ceresNetID}/{ceresNetID}.zip", tempDirectory, console);

        string downloadedFileName = Path.Combine(tempDirectory, ceresNetID + ".onnx");
        File.Move(downloadedFileName, targetFileName);
      }
      catch (Exception ex)
      {
        if (console != null)
        {
          console.WriteException(ex);
        }
        else
        {
          Console.WriteLine($"uci info error downloading {ceresNetID} into {ceresNetDirectory}");
        }
        targetFileName = null; // mark as failure
      }
      finally
      {
        if (tempDirectory != null)
        {
          // Clean up temp directory.
          foreach (string filePath in Directory.GetFiles(tempDirectory))
          {
            File.Delete(filePath);
          }
          Directory.Delete(tempDirectory);
        }

        Console.CancelKeyPress -= (sender, e) => { };
      }

      if (shouldCancel)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Download cancelled.");
        return null;
      }

      return targetFileName;
    }


    /// <summary>
    /// Downloads a file from the specified URL and unpacks it to the specified directory.
    /// </summary>
    /// <param name="downloadID"></param>
    /// <param name="targetDirectory"></param>
    /// <param name="fileUrl"></param>
    /// <param name="directoryPath"></param>
    /// <param name="console"></param>
    /// <returns></returns>
    string DownloadAndUnpackFile(string downloadID, string targetDirectory, string fileUrl, string directoryPath, IAnsiConsole console)
    {
      string localFilePath = Path.Combine(directoryPath, Path.GetFileName(fileUrl));
      using HttpResponseMessage response = client.GetAsync(fileUrl, HttpCompletionOption.ResponseHeadersRead).Result;

      // Surface HTTP errors (e.g. 404 from a wrong/case-mismatched release tag) before they
      // get written to disk and re-emerge as a confusing "Central Directory corrupt" zip error.
      if (!response.IsSuccessStatusCode)
      {
        throw new IOException($"HTTP {(int)response.StatusCode} {response.ReasonPhrase} downloading {fileUrl}");
      }

      long totalBytes = response.Content.Headers.ContentLength ?? 0;

      void DoDownload(string localFilePath, HttpResponseMessage response, ProgressTask task)
      {
        using (Stream stream = response.Content.ReadAsStream())
        using (FileStream fileStream = new FileStream(localFilePath, FileMode.Create))
        {
          byte[] buffer = new byte[1024 * 64];
          int bytesRead;
          while (!shouldCancel &&
                 (bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
          {
            fileStream.Write(buffer, 0, bytesRead);
            task?.Increment(bytesRead);
          }
        }
      }

      if (console == null)
      {
        DoDownload(localFilePath, response, null);
      }
      else
      {
        console.Progress()
            .AutoClear(false)
            .Start(ctx =>
            {
              float sizeMB = MathF.Round(totalBytes / (1024 * 1024), 1);
              ProgressTask task = ctx.AddTask($"[green]Downloading {downloadID} ({sizeMB:F1} mb) to {targetDirectory}  [/]", new ProgressTaskSettings { MaxValue = totalBytes });
              DoDownload(localFilePath, response, task);
            });
      }

      if (!shouldCancel)
      {
        VerifyDownloadedZip(localFilePath, totalBytes, fileUrl);
        ZipFile.ExtractToDirectory(localFilePath, directoryPath);
      }

      return directoryPath;
    }


    /// <summary>
    /// Confirms the file on disk is a plausible zip: matches Content-Length (when known)
    /// and starts with the PK\x03\x04 local-file-header signature. Catches the case where
    /// a redirect/proxy/error page was silently written to disk in place of the real asset.
    /// </summary>
    static void VerifyDownloadedZip(string localFilePath, long expectedBytes, string fileUrl)
    {
      long actualBytes = new FileInfo(localFilePath).Length;
      if (expectedBytes > 0 && actualBytes != expectedBytes)
      {
        throw new IOException($"Truncated download of {fileUrl}: expected {expectedBytes} bytes, got {actualBytes}");
      }

      // Smallest legal zip is 22 bytes (empty archive's End-of-Central-Directory record);
      // a real asset will be much larger, so anything tiny here is an error page or redirect body.
      if (actualBytes < 22)
      {
        throw new IOException($"Download of {fileUrl} produced a {actualBytes}-byte file (not a zip)");
      }

      Span<byte> magic = stackalloc byte[4];
      using (FileStream fs = new FileStream(localFilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
      {
        if (fs.Read(magic) != 4 || magic[0] != 0x50 || magic[1] != 0x4B || magic[2] != 0x03 || magic[3] != 0x04)
        {
          throw new IOException($"Download of {fileUrl} is not a zip file (missing PK\\x03\\x04 header)");
        }
      }
    }

    async Task<string> DownloadInfoFile(string infoUrl, IAnsiConsole console)
    {
      HttpResponseMessage response = await client.GetAsync(infoUrl);
      string content = await response.Content.ReadAsStringAsync();
      console.WriteLine(content);
      return content;
    }
  }
}
