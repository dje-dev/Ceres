#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using Directives

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

#endregion

namespace Ceres.Base.OperatingSystem.Windows
{
  /// <summary>
  /// Interop with Windows APIs related to memory and processor configuration.
  /// 
  /// Adapted from code by Michael Vanhoutte from
  /// http://blogs.adamsoftware.net/Engine/DeterminingthenumberofphysicalCPUsonWindows.aspx
  ///
  /// </summary>
  [SupportedOSPlatform("windows")]
  public static class WindowsHardware
  {
    private const int ERROR_INSUFFICIENT_BUFFER = 122;

    private enum PROCESSOR_CACHE_TYPE
    {
      /// <summary>
      /// The cache is unified.
      /// </summary>
      UnifiedCache = 0,

      /// <summary>
      /// Instruction cache is for processor instructions.
      /// </summary>
      InstructionCache = 1,

      /// <summary>
      /// The cache is for data.
      /// </summary>
      DataCache = 2,

      /// <summary>
      /// The cache is for traces.
      /// </summary>
      TraceCache = 3
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CACHE_DESCRIPTOR
    {
      byte Level;
      byte Associativity;
      UInt16 LineSize;
      UInt32 Size;
      [MarshalAs(UnmanagedType.U4)]
      PROCESSOR_CACHE_TYPE Type;
    }

    public enum RelationProcessorCore
    {
      /// <summary>
      /// The specified logical processors share a 
      /// single processor core.    
      /// </summary>
      RelationProcessorCore = 0,

      /// <summary>
      /// The specified logical processors are part 
      /// of the same NUMA node.
      /// </summary>
      RelationNumaNode = 1,

      /// <summary>
      /// The specified logical processors  share a cache. 
      /// Windows Server 2003:  This value is not supported 
      /// until Windows Server 2003 SP1 and Windows XP 
      /// Professional x64 Edition.
      /// </summary>
      RelationCache = 2,

      /// <summary>
      /// The specified logical processors share a physical 
      /// package (a single package socketed or soldered 
      /// onto a motherboard may contain multiple processor 
      /// cores or threads, each of which is treated as a 
      /// separate processor by the operating system). 
      /// Windows Server 2003:  This value is not 
      /// supported until Windows Vista.
      /// </summary>
      RelationProcessorPackage = 3
    }


    [StructLayout(LayoutKind.Explicit)]
    public struct SYSTEM_LOGICAL_PROCESSOR_INFORMATIONx64
    {
      [FieldOffset(0)]
      public uint ProcessorMask;
      [FieldOffset(8), MarshalAs(UnmanagedType.U4)]
      public RelationProcessorCore Relationship;
      [FieldOffset(12)]
      public byte Flags;
      [FieldOffset(12)]
      public CACHE_DESCRIPTOR Cache;
      [FieldOffset(12)]
      public UInt32 NodeNumber;
      [FieldOffset(12)]
      public UInt64 Reserved1;
      [FieldOffset(20)]
      public UInt64 Reserved2;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GetLogicalProcessorInformation([Out] SYSTEM_LOGICAL_PROCESSOR_INFORMATIONx64[] infos, ref uint infoSize);

    public class ProcessorInfo
    {
      public ProcessorInfo(RelationProcessorCore relationShip, byte flags, uint processorMask)
      {
        Relationship = relationShip;
        Flags = flags;
        ProcessorMask = processorMask;
      }

      public RelationProcessorCore Relationship { get; private set; }
      public byte Flags { get; private set; }
      public uint ProcessorMask { get; private set; }

      public override string ToString()
      {
        return "<ProcessorInfo " + Relationship + " " + ProcessorMask + " " + Flags + ">";
      }
    }

    /// <summary>
    /// Returns the number of CPU sockets (NUMA nodes) exposed by the hardware.
    /// </summary>
    public static int NumCPUSockets
    {
      get
      {
        int socketCount = 0;
        foreach (var pi in GetProcessorInfo64())
          if (pi.Relationship == RelationProcessorCore.RelationNumaNode)
            socketCount++;
        return socketCount;
      }
    }

    public static List<ProcessorInfo> GetProcessorInfo64()
    {
      uint iReturnLength = 0;
      SYSTEM_LOGICAL_PROCESSOR_INFORMATIONx64[] oDummy = null;
      bool bResult = GetLogicalProcessorInformation(oDummy, ref iReturnLength);
      if (bResult)
        throw Fail("GetLogicalProcessorInformation failed.", "x64");

      int iError = Marshal.GetLastWin32Error();
      if (iError != ERROR_INSUFFICIENT_BUFFER)
        throw Fail("Insufficient space in the buffer.", "x64", iError.ToString());

      uint iBaseSize = (uint)Marshal.SizeOf(typeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATIONx64));
      uint iNumberOfElements = iReturnLength / iBaseSize;
      SYSTEM_LOGICAL_PROCESSOR_INFORMATIONx64[] oData = new SYSTEM_LOGICAL_PROCESSOR_INFORMATIONx64[iNumberOfElements];
      uint iAllocatedSize = iNumberOfElements * iBaseSize;
      if (!GetLogicalProcessorInformation(oData, ref iAllocatedSize))
        throw Fail("GetLogicalProcessorInformation failed", "x64", Marshal.GetLastWin32Error().ToString());

      // Converting the data to a list that we can easily interpret.
      List<ProcessorInfo> oList = new List<ProcessorInfo>();
      foreach (SYSTEM_LOGICAL_PROCESSOR_INFORMATIONx64 oInfo in oData)
        oList.Add(new ProcessorInfo(oInfo.Relationship, oInfo.Flags, oInfo.ProcessorMask));
      return oList;
    }

    private static Exception Fail(params string[] data)
    {
      return new NotSupportedException("GetPhysicalProcessorCount unexpectedly failed " +
                                        "(" + String.Join(", ", data) + ")");
    }


    /// <summary>
    /// Returns number of physical threads (not including logical hyperthreads).
    /// 
    /// NOTE: It has been pointed out that one really should look at the processor
    ///       affinity mask and see how many over how many processors this process
    ///       is actually eligible to run. See "Don't rely on Environment.ProcessorCount" blog entry
    /// </summary>
    /// <returns></returns>
    public static int GetPhysicalThreadCount()
    {
      try
      {
        return TryGetPhysicalThreadCount();
      }
      catch (Exception )
      {
        // Fall back to value returned by Enviroment (which may represent virtual rather than physical processors)
        return System.Environment.ProcessorCount;
      }
    }


    static int TryGetPhysicalThreadCount()
    {
      // Getting a list of processor information
      List<ProcessorInfo> processors = GetProcessorInfo64();

      // The list will basically contain something like this at this point:
      //
      // E.g. for a 2 x single core
      // Relationship              Flags      ProcessorMask
      // ---------------------------------------------------------
      // RelationProcessorCore     0          1
      // RelationProcessorCore     0          2
      // RelationNumaNode          0          3
      //
      // E.g. for a 2 x dual core
      // Relationship              Flags      ProcessorMask
      // ---------------------------------------------------------
      // RelationProcessorCore     1          5
      // RelationProcessorCore     1          10
      // RelationNumaNode          0          15
      //
      // E.g. for a 1 x quad core
      // Relationship              Flags      ProcessorMask
      // ---------------------------------------------------------
      // RelationProcessorCore     1          15
      // RelationNumaNode          0          15
      //
      // E.g. for a 1 x dual core
      // Relationship              Flags      ProcessorMask  
      // ---------------------------------------------------------
      // RelationProcessorCore     0          1              
      // RelationCache             1          1              
      // RelationCache             1          1              
      // RelationProcessorPackage  0          3              
      // RelationProcessorCore     0          2              
      // RelationCache             1          2              
      // RelationCache             1          2              
      // RelationCache             2          3              
      // RelationNumaNode          0          3
      // 
      // Vista or higher will return one RelationProcessorPackage 
      // line per socket. On other operating systems we need to 
      // interpret the RelationProcessorCore lines.
      //
      // More information:
      // http://msdn2.microsoft.com/en-us/library/ms683194(VS.85).aspx
      // http://msdn2.microsoft.com/en-us/library/ms686694(VS.85).aspx

      return processors.Select(i => i.Relationship == RelationProcessorCore.RelationProcessorCore).Count();
#if NOT
        // First counting the number of RelationProcessorPackage lines
        int iCount = oList.Select(i => i.Relationship == RelationProcessorCore.RelationProcessorPackage).Count;
        if (iCount > 0) return iCount;

        // Now we're going to use the information in RelationProcessorCore.
        iCount = 0;
        foreach (ProcessorInfo oItem in oList)
        {
          if (oItem.Relationship ==
            RelationProcessorCore.RelationProcessorCore)
            iCount++;
        }

        if (iCount > 0)
          return iCount;

        throw Fail("No cpus have been detected.");
#endif

    }
  }
}

