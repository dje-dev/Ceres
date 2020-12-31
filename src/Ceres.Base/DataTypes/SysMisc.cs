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
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

using System.Runtime.InteropServices;
using System.Runtime.Serialization.Formatters.Binary;
using System.Runtime.CompilerServices;


#endregion

namespace Ceres.Base.DataType
{
  /// <summary>
  /// Static helper methods relating to low-level system operations.
  /// </summary>
  public static class SysMisc
  {
    public static unsafe void WriteSpanToStream<T>(Stream stream, Span<T> items) where T : unmanaged
    {
      void* pointer = Unsafe.AsPointer(ref items[0]);
      Span<byte> span = new Span<byte>(pointer, sizeof(T) * items.Length);
      stream.Write(span);
    }


    public static unsafe void ReadStreamIntoSpan<T>(Stream stream, Span<T> items, int numItems) where T : unmanaged
    {
      if (items.Length < numItems) throw new ArgumentOutOfRangeException("Span too small to receive items");

      void* pointer = Unsafe.AsPointer(ref items[0]);
      Span<byte> byteSpan = new Span<byte>(pointer, sizeof(T) * numItems);
      stream.Read(byteSpan);
    }


    public static void WriteSpanToFile<T> (string FN, Span<T> span) where T : unmanaged
    {
      FileStream stream = File.Create(FN);
      try
      {
        WriteSpanToStream<T>(stream, span);
      }
      finally
      {
        stream.Close();
      }
    }

    public static long ReadFileIntoSpan<T>(string FN, Span<T> span) where T : unmanaged
    {
      if (!File.Exists(FN)) throw new Exception("file not found " + FN);

      long numBytes = new FileInfo(FN).Length;
      if (numBytes == 0) throw new Exception("File is empty " + FN);

      long leftoverBytes = numBytes % Marshal.SizeOf<T>();
      if (leftoverBytes != 0) throw new Exception("File size inconsistent with integral number of items");

      long numItems = numBytes / Marshal.SizeOf<T>();
      FileStream stream = new FileStream(FN, FileMode.Open);
      try
      {
        ReadStreamIntoSpan(stream, span, (int)numItems);
      }
      finally
      {
        stream.Close();
      }

      return numItems;
    }


    public static void WriteObj(string FN, object obj)
    {
      FileStream stream = File.Create(FN);
      BinaryFormatter formatter = new BinaryFormatter();
      formatter.Serialize(stream, obj);
      stream.Close();
    }


    public static T ReadObj<T>(string FN)
    {
      byte[] allBytes = File.ReadAllBytes(FN);
      BinaryFormatter formatter = new BinaryFormatter();
      object obj = formatter.Deserialize(new FileStream(FN, FileMode.Open, FileAccess.Read));
      return (T)obj;
    }


    [DllImport("kernel32.dll")]
    public static extern int DeviceIoControl(IntPtr hDevice, int
        dwIoControlCode, ref short lpInBuffer, int nInBufferSize, IntPtr
        lpOutBuffer, int nOutBufferSize, ref int lpBytesReturned, IntPtr
        lpOverlapped);


    public static void NTFSCompressFile(FileStream f)
    {
      int lpBytesReturned = 0;
      int FSCTL_SET_COMPRESSION = 0x9C040;
      short COMPRESSION_FORMAT_DEFAULT = 1; // default

      int result = DeviceIoControl(f.Handle, FSCTL_SET_COMPRESSION, ref COMPRESSION_FORMAT_DEFAULT,
                                   2 /*sizeof(short)*/, IntPtr.Zero, 0, ref lpBytesReturned, IntPtr.Zero);
    }


    public static void NTFSCompressFile(string fileName)
    {
      FileStream f = File.Open(fileName, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
      NTFSCompressFile(f);
      f.Close();
  }

    public static void RunConcurrent(Action action, int numConcurrent)
    {
      List<Task> tasks = new List<Task>();
      for (int i = 0; i < numConcurrent; i++)
        tasks.Add(Task.Factory.StartNew(action));

      Task.WaitAll(tasks.ToArray());
    }


    public static string ShellExecute(this string path, string command, TextWriter writer, params string[] arguments)
    {
      using (var process = Process.Start(new ProcessStartInfo
      {
        WorkingDirectory = path,
        FileName = command,
        Arguments = string.Join(" ", arguments),
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true
      }))
      {
        using (process.StandardOutput)
        {
          writer.WriteLine(process.StandardOutput.ReadToEnd());
        }
        using (process.StandardError)
        {
          writer.WriteLine(process.StandardError.ReadToEnd());
        }
      }

      return path;
    }


    public static string CreateTmpFile(string suffix = null)
    {
      string fileName = string.Empty;

      // Get the full name of the newly created Temporary file (0 byte file created)
      fileName = Path.GetTempFileName();
      if (suffix != null) fileName = fileName + "." + suffix;

      // Create a FileInfo object to set the file's attributes
      FileInfo fileInfo = new FileInfo(fileName);
      //fileInfo.Attributes = FileAttributes.Temporary;

      return fileName;
    }


  }
}
