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
using System.Reflection;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Static helper methods for working with strings.
  /// </summary>
  public static class ObjUtils
  {
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
            DumpArrray(sb, (Array)val);
          }
          else
          {
            sb.AppendLine(val == null ? "(null)" : val.ToString());
          }
        }
      }

      return differentOnly && countDifferent == 0 ? null : sb.ToString();
    }


    static void DumpArrray(StringBuilder sb, Array array)
    {
      sb.Append("{");
      for (int i = 0; i < array.Length; i++)
      {
        if (i > 0) sb.Append(",");
        sb.Append(array.GetValue(i));
      }
      sb.Append("}");
    }
  }
}
