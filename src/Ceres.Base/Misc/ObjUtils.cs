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
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Static helper methods for working with objects.
  /// </summary>
  public static class ObjUtils
  {

  public static T DeepClone<T>(T a)
  {
#if AVOID_BINARY_SERIALIZATION
    throw new Exception("WARNING: Not yet working, causes strange errors on Linux in TournamentManager with AccessDenied on Console.Out");
    return (T)ObjUtilsCopy.Copy(a);
#else
      using (MemoryStream stream = new MemoryStream())
      {
        BinaryFormatter formatter = new BinaryFormatter();
        formatter.Serialize(stream, a);
        stream.Position = 0;
        return (T)formatter.Deserialize(stream);
      }
#endif
    }

    /// <summary>
    /// Interprets the given byte array as an array of structures of type T and copies into a specified output buffer of type T.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="rawBuffer"></param>
    /// <param name="buffer"></param>
    /// <param name="bytesRead"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    /// <exception cref="NotImplementedException"></exception>
    public static unsafe int CopyBytesIntoStructArray<T>(byte[] rawBuffer, T[] buffer, int bytesRead) where T : unmanaged
    {
      int itemSize = Marshal.SizeOf(typeof(T));
      if (bytesRead == 0 || bytesRead % itemSize != 0)
      {
        throw new Exception($"ReadFromStream returned {bytesRead} which is not a multiple of intended structure size");
      }

      // Determine and validate number of items.
      int numItems = bytesRead / itemSize;
      if (numItems > buffer.Length)
      {
        throw new NotImplementedException($"buffer sized {buffer.Length} too small for required {numItems}");
      }

      // Copy items into target buffer.
      fixed (byte* source = &rawBuffer[0])
      {
        fixed (T* target = &buffer[0])
        {
          Unsafe.CopyBlock(target, source, (uint)bytesRead);
        }
      }

      return numItems;
    }


    public static string FieldValuesDumpString<T>(object obj, object comparisonValue, bool differentOnly = false) where T : class
    {
      StringBuilder sb = new StringBuilder();
      sb.AppendLine(typeof(T).Name);

      int countDifferent = 0;
      // TODO: someday dump non public
      foreach (FieldInfo propInfo in obj.GetType().GetFields())
      {
        if (propInfo.FieldType.IsValueType)
        {
          object val = propInfo.GetValue(obj);
          object valDefault = propInfo.GetValue(comparisonValue);

          bool different = ((val == null) != (valDefault == null)) || !val.Equals(valDefault);
          if (different)
          {
            countDifferent++;
          }

          if (!different && differentOnly) continue;

          sb.Append(different ? "* " : "  ");
          sb.Append($"{propInfo.Name,-60}");

          if (propInfo.FieldType.IsArray && val != null)
          {
            DumpArray(sb, (Array)val);
          }
          else
          {
            sb.AppendLine(val == null ? "(null)" : val.ToString());
          }
        }
      }

      return differentOnly && countDifferent == 0 ? null : sb.ToString();
    }


    /// <summary>
    /// Dumps the fields of two objects side by side.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="struct1"></param>
    /// <param name="struct2"></param>
    public static void CompareAndPrintObjectFields<T>(T struct1, T struct2)
    {
      Console.WriteLine("\r\nCOMPARE OBJECTS " + typeof(T).Name);
      foreach (FieldInfo field in typeof(T).GetFields(BindingFlags.Public | BindingFlags.Instance))
      {
        string fieldName = field.Name;
        object value1 = field.GetValue(struct1);
        object value2 = field.GetValue(struct2);

        Console.WriteLine($"{fieldName,20}:  {value1} vs {value2}");
      }
    }


    static void DumpArray(StringBuilder sb, Array array)
    {
      sb.Append("{");
      for (int i = 0; i < array.Length; i++)
      {
        if (i > 0) sb.Append(",");
        sb.Append(array.GetValue(i));
      }
      sb.Append("}");
    }

    public static void Shuffle<T>(IList<T> list)
    {
      Random rand = new(System.Environment.TickCount);

      int n = list.Count;
      while (n > 1)
      {
        n--;
        int k = rand.Next(n + 1);
        T value = list[k];
        list[k] = list[n];
        list[n] = value;
      }
    }


  }
}
