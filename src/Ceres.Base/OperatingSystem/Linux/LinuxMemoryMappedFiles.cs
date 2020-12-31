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
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Threading;

#endregion

/// <summary>
/// 
/// Similar: https://github.com/tmds/Tmds.LibC/tree/master/src/Sources/linux.common
/// </summary>
namespace Ceres.Base.OperatingSystem.Linux
{

  /// <summary>
  /// This might not eventually be needed but serves as good sample code until we get it working).
  /// 
  /// https://stackoverflow.com/questions/27842300/c-to-c-sharp-mono-memory-mapped-files-shared-memory-in-linux
  /// </summary>
  public static  class LinuxMemoryMappedFiles
  {
    const string filepath = "/tmp/sharedfile";

    // --------------------------------------------------------------------------------------------
    public static void TestCPP()
    {

#if NOT
    int fd;
    int index;
    char *data;
    const char *filepath = "/tmp/sharedfile";

    if ((fd = open(filepath, O_CREAT|O_RDWR, (mode_t)00700)) == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    data = mmap(NULL, 12288, PROT_WRITE|PROT_READ, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }


    for (index= 0; index < 200; index++) {
        data[index] = 'G';
    } 

    sleep(10);

    // We must see 'Goose' at the beginning of memory-mapped file.
    for (index= 0; index < 200; index++) {
        fprintf(stdout, "%c", data[index]);
    }

    for (index= 0; index < 200; index++) {
        data[index] = 'H';
    }

    if (msync(data, 12288, MS_SYNC) == -1) {
        perror("Error sync to disk");
    } 

    if (munmap(data, 12288) == -1) {
        close(fd);
        perror("Error un-mmapping");
        exit(EXIT_FAILURE);
    }

    close(fd);
#endif
    }

    // --------------------------------------------------------------------------------------------
    public static void TestCSharp()
    {
      using (var mmf = MemoryMappedFile.CreateFromFile("/tmp/sharedfile", FileMode.OpenOrCreate, "/tmp/sharedfile"))
      {
        using (var stream = mmf.CreateViewStream())
        {
          // 1. C program, filled memory-mapped file with the 'G' character (200 characters)
          var data = stream.ReadByte();
          while (data != -1)
          {
            Console.WriteLine((char)data);
            data = stream.ReadByte();
          }

          // 2. We write "Goose" at the beginning of memory-mapped file.
          stream.Position = 0;
          var buffer = new byte[] { 0x47, 0x6F, 0x6F, 0x73, 0x65 };
          stream.Write(buffer, 0, 5);

          Thread.Sleep(20000);

          // 3. C program, filled memory-mapped file with the 'H' character (200 characters)
          stream.Position = 0;
          data = stream.ReadByte();
          while (data != -1)
          {
            Console.WriteLine((char)data);
            data = stream.ReadByte();
          }
        }
      }
    }
  }
}