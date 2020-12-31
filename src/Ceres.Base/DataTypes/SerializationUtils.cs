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
using System.Runtime.InteropServices;
using ProtoBuf;

#endregion

namespace Ceres.Base.DataType
{
  /// <summary>
  /// Static utility methods related to binary serialization.
  /// </summary>
  public static class SerializationUtils
  {
    #region Protobuf

    /// <summary>
    /// Serializes a protobuf from a specified type.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="record"></param>
    /// <returns></returns>
    public static byte[] ProtoSerialize<T>(T record) where T : class
    {
      if (null == record) return null;

      try
      {
        using (var stream = new MemoryStream())
        {
          Serializer.Serialize(stream, record);
          return stream.ToArray();
        }
      }
      catch
      {
        // Log error
        throw;
      }
    }


    /// <summary>
    /// Deserializes a protobuf into a specified type.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="data"></param>
    /// <returns></returns>
    public static T ProtoDeserialize<T>(byte[] data) where T : class
    {
      if (null == data) return null;

      try
      {
        using (var stream = new MemoryStream(data))
        {
          return Serializer.Deserialize<T>(stream);
        }
      }
      catch
      {
        // Log error
        throw;
      }
    }

    #endregion

    #region Serialize structures

   
    /// <summary>
    /// Serializes an arbitrary unamnaged type into an array of bytes.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="s"></param>
    /// <returns></returns>
    public static byte[] Serialize<T>(T s) where T : unmanaged
    {
      int size = Marshal.SizeOf(typeof(T));
      byte[] array = new byte[size];
      IntPtr ptr = Marshal.AllocHGlobal(size);
      Marshal.StructureToPtr(s, ptr, true);
      Marshal.Copy(ptr, array, 0, size);
      Marshal.FreeHGlobal(ptr);
      return array;
    }


    /// <summary>
    /// Deserializes an arbitrary unamnaged type from an array of bytes.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="array"></param>
    /// <returns></returns>
    public static T Deserialize<T>(byte[] array) where T : unmanaged
    {
      int size = Marshal.SizeOf(typeof(T));
      IntPtr ptr = Marshal.AllocHGlobal(size);
      Marshal.Copy(array, 0, ptr, size);
      T s = (T)Marshal.PtrToStructure(ptr, typeof(T));
      Marshal.FreeHGlobal(ptr);
      return s;
    }

    #endregion

    #region Serialize from bytes

    /// <summary>
    /// Converts between an array of objects and byte array.
    /// 
    /// NOTE: Often it will be much faster to use Marshal.Cast instead
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="array"></param>
    /// <param name="numToSerialize"></param>
    /// <returns></returns>
    public static byte[] SerializeArray<T>(T[] array, int numToSerialize = -1) where T : struct
    {
      int unmananagedSize = Marshal.SizeOf(typeof(T));

      long numBytesLong;
      if (numToSerialize == -1)
        numBytesLong = (long)array.Length * (long)unmananagedSize;
      else
      {
        if (numToSerialize > array.Length) throw new Exception("unexpected length");
        numBytesLong = (long)numToSerialize * (long)unmananagedSize;
      }

      if (numBytesLong > int.MaxValue) throw new ArgumentException($"Array size { numBytesLong} exceeds max allowable bytes of {int.MaxValue}");

      byte[] bytes = new byte[(int)numBytesLong];

      GCHandle handleDest = default;
      try
      {
        handleDest = GCHandle.Alloc(array, GCHandleType.Pinned);
        Marshal.Copy(handleDest.AddrOfPinnedObject(), bytes, 0, (int)numBytesLong);
      }
      finally
      {
        if (handleDest.IsAllocated)
          handleDest.Free();
      }

      return bytes;
    }


    /// <summary>
    /// Converts between an array of objects and byte array.
    /// 
    /// NOTE: Often it will be much faster to use Marshal.Cast instead
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="bytes"></param>
    /// <param name="lengthInBytes"></param>
    /// <param name="buffer"></param>
    /// <returns></returns>
    public static int DeSerializeArrayIntoBuffer<T>(byte[] bytes, int lengthInBytes, ref T[] buffer) where T : struct
    {
      int unmananagedSize = Marshal.SizeOf(typeof(T));

      // Integrity check on the size of the bytes
      int modulo = lengthInBytes % unmananagedSize;
      if (modulo != 0)
        throw new Exception("Bytes length not evenly divisble by structure size");

      int numItems = lengthInBytes / unmananagedSize;
      if (numItems > buffer.Length) throw new Exception("Deseriaization buffer size insufficient for data");

      GCHandle handleDest = default;
      try
      {
        handleDest = GCHandle.Alloc(buffer, GCHandleType.Pinned);
        Marshal.Copy(bytes, 0, handleDest.AddrOfPinnedObject(), lengthInBytes);
      }
      finally
      {
        if (handleDest.IsAllocated)
          handleDest.Free();
      }
      return numItems;
    }


    /// <summary>
    /// Converts between an array of objects and byte array.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="bytes"></param>
    /// <param name="numBytesToDeserialize"></param>
    /// <returns></returns>
    public static T[] DeSerializeArray<T>(byte[] bytes, int numBytesToDeserialize) where T : struct
    {
      int unmananagedSize = Marshal.SizeOf(typeof(T));

      int numItems = numBytesToDeserialize / unmananagedSize;
      if (numBytesToDeserialize % unmananagedSize != 0) throw new Exception("Wrong sized buffer");

      T[] newArray = new T[numItems];

      GCHandle handleDest = default;
      try
      {
        handleDest = GCHandle.Alloc(newArray, GCHandleType.Pinned);
        Marshal.Copy(bytes, 0, handleDest.AddrOfPinnedObject(), numBytesToDeserialize);
      }
      finally
      {
        if (handleDest.IsAllocated)
          handleDest.Free();
      }
      return newArray;
    }

    #endregion

  }
}
