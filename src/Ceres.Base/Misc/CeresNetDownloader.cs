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
      HttpResponseMessage response = client.GetAsync(fileUrl, HttpCompletionOption.ResponseHeadersRead).Result;
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
        ZipFile.ExtractToDirectory(localFilePath, directoryPath);
      }

      return directoryPath;
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



#if NOT
  public static class GuidedSearch
  {
    public static void Test()
    {
      string ceresJSONPath = @"c:\dev\ceres\artifacts\release\net8.0\Ceres.json";
      Console.WriteLine("Loading " + ceresJSONPath);
      CeresUserSettingsManager.LoadFromFile(ceresJSONPath);


      //      NNEvaluator def = NNEvaluatorDef.FromSpecification("703810", "GPU:0").ToEvaluator();
      NNEvaluatorDef spec = NNEvaluatorDef.FromSpecification("703810", "GPU:0");


      GameEngineCeresInProcess engineCeres = new GameEngineCeresInProcess("Ceres-ZZ", spec, null,
        new ParamsSearch()
        {
        },
        new ParamsSelect()
        {
        });


      const string FN = @"c:\temp\ceres\match_TOURN_Ceres1_Ceres2_638425960050089404.pgn";
      PGNFileEnumerator pgn = new(FN);
      int count = 0;
      foreach (PositionWithHistory pp in pgn.EnumeratePositionWithHistory())
      {
        engineCeres.ResetGame();
        count++;

        GameEngineSearchResultCeres search = engineCeres.SearchCeres(pp, SearchLimit.NodesPerMove(1_000));


        MCTSTree tree = search.Search.Manager.Context.Tree;
        for (int i = 2; i <= 9; i += 2)
        {
          //          MCTSCheckPV.PVCheckResult checkResult = MCTSCheckPVAll.Check(search.Search, null, i);

        }


        {
          if (count % 1000 == 0)
          {
            //            Console.WriteLine("**** " + count + "  " + (float)MCTSCheckPVAll.numSearchExtended / MCTSCheckPVAll.numBaseSearched +
            //              " " + MCTSCheckPVAll.numBlundersFound + " " + MCTSCheckPVAll.sumBlunders);
          }
          if (count > 5_000) System.Environment.Exit(3);
        }

#endif