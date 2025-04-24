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
    /// <summary>
    /// Creates a deep clone using binary serialization (BinaryFormatter).
    /// 
    /// Note that Microsoft discourage use of this library due to security concerns.
    /// Those concerns only apply if the source of the serialization data is coming from
    /// an unknown/untrusted source (such as file on disk that might have been manipulated).
    /// However the Ceres usage is all "in process" and free of any such concerns.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="a"></param>
    /// <returns></returns>
    public static T DeepClone<T>(T a)
    {
      using (MemoryStream stream = new MemoryStream())
      {
        BinaryFormatter formatter = new BinaryFormatter();
        formatter.Serialize(stream, a);
        stream.Position = 0;
        return (T)formatter.Deserialize(stream);
      }
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


    /// <summary>
    /// Dumps the field values of an object of type T into a string (via reflection).
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="obj"></param>
    /// <param name="comparisonValue"></param>
    /// <param name="differentOnly"></param>
    /// <returns></returns>
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
    /// Dumps the field values of a struct of type T into a string (via reflection).
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="obj"></param>
    /// <param name="comparisonValue"></param>
    /// <param name="differentOnly"></param>
    /// <returns></returns>
    public static string FieldValuesDumpString<T>(T obj, T comparisonValue, bool differentOnly = false) where T : struct
    {
      StringBuilder sb = new StringBuilder();
      sb.AppendLine(typeof(T).Name);

      int countDifferent = 0;
      // Loop over the fields declared in T.
      foreach (FieldInfo fieldInfo in typeof(T).GetFields())
      {
        object val = fieldInfo.GetValue(obj);
        object compVal = fieldInfo.GetValue(comparisonValue);

        // For value types the null-check is usually redundant,
        // but we include it for safety (in case a field is a reference type).
        bool different = ((val == null) != (compVal == null)) || (val != null && !val.Equals(compVal));
        if (different)
        {
          countDifferent++;
        }

        if (!different && differentOnly)
          continue;

        sb.Append(different ? "* " : "  ");
        sb.Append($"{fieldInfo.Name,-60}");

        if (fieldInfo.FieldType.IsArray && val != null)
        {
          DumpArray(sb, (Array)val);
        }
        else
        {
          sb.AppendLine(val == null ? "(null)" : val.ToString());
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
    public static void CompareAndPrintObjectFields<T>(in T struct1, in T struct2)
    {
      int sumBytes = 0;
      Console.WriteLine("\r\nCOMPARE OBJECTS " + typeof(T).Name);
      foreach (FieldInfo field in typeof(T).GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
      {
        int bytes;

        if (field.FieldType.IsEnum)
        {
          // Get the underlying type of the enum
          Type underlyingType = Enum.GetUnderlyingType(field.FieldType);

          // Get the size of the underlying type
          bytes = Marshal.SizeOf(underlyingType);
        }
        else
        {
          // Get the size of the field type
          bytes = Marshal.SizeOf(field.FieldType);
        }

        sumBytes += bytes;
        string fieldName = field.Name;
        object value1 = field.GetValue(struct1);
        object value2 = field.GetValue(struct2);

        Console.WriteLine($"  {bytes,4}   {fieldName,20}:  {value1} vs {value2}");
      }
      Console.WriteLine($"Total bytes: {sumBytes}");
    }


    /// <summary>
    /// Returns string summarizing the differences between fields/propery values in two structs.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="t1"></param>
    /// <param name="t2"></param>
    /// <returns></returns>
    public static string StructDiffs<T>(T t1, T t2) where T : struct
    {
      StringBuilder differences = new StringBuilder();
      Type type = typeof(T);

      // Process fields.
      ProcessMembers(differences, type.GetFields(BindingFlags.Public | BindingFlags.Instance), t1, t2);

      // Process properties.
      ProcessMembers(differences, type.GetProperties(BindingFlags.Public | BindingFlags.Instance), t1, t2);

      return differences.ToString();
    }


    private static void ProcessMembers<T>(StringBuilder differences, IEnumerable<MemberInfo> members, T t1, T t2) where T : struct
    {
      foreach (var member in members)
      {
        object value1, value2;

        switch (member)
        {
          case FieldInfo field:
            value1 = field.GetValue(t1);
            value2 = field.GetValue(t2);
            break;

          case PropertyInfo property:
            if (property.GetMethod != null) // Ensure the property is readable
            {
              value1 = property.GetValue(t1);
              value2 = property.GetValue(t2);
              break;
            }
            continue;

          default:
            continue;
        }

        if (!object.Equals(value1, value2))
        {
          if (differences.Length > 0)
            differences.Append("  ");

          differences.AppendFormat("{0}: {1} --> {2}.", member.Name, value1, value2);
        }
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
