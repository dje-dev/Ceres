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

using System.IO;

namespace Ceres.Chess.NNFiles
{
  /// <summary>
  /// Descriptive information about the a file containing 
  /// the weights/structure of a file for a value/policy neural network.
  /// </summary>
  public interface INNWeightsFileInfo
  {
    /// <summary>
    /// Description of generator of weights file (e.g. Leela Chess Zero).
    /// </summary>
    public string GeneratorDescription { get; }

    /// <summary>
    /// Unique network ID (for example typically sequential numeric values such as "32930").
    /// </summary>
    public string NetworkID { get; }

    /// <summary>
    /// Path and name of file containing weights.
    /// </summary>
    public string FileName { get; }

    /// <summary>
    /// Number of blocks (layers) in the convolution section of the network.
    /// </summary>
    public int NumBlocks { get; }

    /// <summary>
    /// Number of convolutional filters.
    /// </summary>
    public int NumFilters { get; }

    /// <summary>
    /// If the network contains a WDL (win/draw/loss) value head.
    /// </summary>
    public bool IsWDL { get; }

    /// <summary>
    /// If the network contains an MLH (moves left head).
    /// </summary>
    public bool HasMovesLeft { get; }

    /// <summary>
    /// Information about the underlying file from the operating system.
    /// </summary>
    public FileInfo FileInfo { get; }


    /// <summary>
    /// Returns a short descriptive string.
    /// </summary>
    public string ShortStr
    {
      get
      {
        string headsString = (IsWDL ? "WDL " : "") + (HasMovesLeft ? "MLH " : "");
        if (string.Compare(NetworkID, FileName, true) == 0)
        {
          // Network ID is already the full file name
          return $"{NumBlocks}x{NumFilters} {headsString} from {FileName}";
        }
        else
        {
          return $"{NetworkID}: {NumBlocks}x{NumFilters} {headsString} from {FileName}";
        }
      }
    }

    public string ONNXFileName
    {
      // TODO: eliminate hardcoding
      get
      {
        return NetworkID;
#if NOT

        string baseName = FileName.ToLower().Replace(".pb.gz", ".onnx");
        baseName = baseName.Replace(".pb", ".onnx");
        baseName = baseName.Replace("weights_", "");
        baseName = baseName.Replace(@"d:\weights\lczero.org", @"d:\converted");
        return Path.Combine(".", baseName);
#endif
      }
    }
  }
}
