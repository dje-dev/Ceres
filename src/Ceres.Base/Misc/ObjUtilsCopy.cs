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
using System.Reflection;

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Helper class to facilitate  deep cloning of types.
  /// Based on work of Alex Burtsev (https://stackoverflow.com/questions/129389/how-do-you-do-a-deep-copy-of-an-object-in-net).
  /// </summary>
  internal static class ObjUtilsCopy
  {
    private static readonly MethodInfo CloneMethod = typeof(Object).GetMethod("MemberwiseClone", BindingFlags.NonPublic | BindingFlags.Instance);

    internal static bool IsPrimitive(Type type)
    {
      if (type == typeof(String)) return true;
      return (type.IsValueType & type.IsPrimitive);
    }

    internal static Object Copy(Object originalObject)
    {
      return InternalCopy(originalObject, new Dictionary<Object, Object>(new ReferenceEqualityComparer()));
    }

    private static Object InternalCopy(Object originalObject, IDictionary<Object, Object> visited)
    {
      if (originalObject == null) return null;
      var typeToReflect = originalObject.GetType();
      if (IsPrimitive(typeToReflect)) return originalObject;
      if (visited.ContainsKey(originalObject)) return visited[originalObject];
      if (typeof(Delegate).IsAssignableFrom(typeToReflect)) return null;
      var cloneObject = CloneMethod.Invoke(originalObject, null);
      if (typeToReflect.IsArray)
      {
        var arrayType = typeToReflect.GetElementType();
        if (IsPrimitive(arrayType) == false)
        {
          Array clonedArray = (Array)cloneObject;
          ArrayExtensions.ForEach(clonedArray, (array, indices) => array.SetValue(InternalCopy(clonedArray.GetValue(indices), visited), indices));
        }

      }
      visited.Add(originalObject, cloneObject);
      CopyFields(originalObject, visited, cloneObject, typeToReflect);
      RecursiveCopyBaseTypePrivateFields(originalObject, visited, cloneObject, typeToReflect);
      return cloneObject;
    }

    private static void RecursiveCopyBaseTypePrivateFields(object originalObject, IDictionary<object, object> visited, object cloneObject, Type typeToReflect)
    {
      if (typeToReflect.BaseType != null)
      {
        RecursiveCopyBaseTypePrivateFields(originalObject, visited, cloneObject, typeToReflect.BaseType);
        CopyFields(originalObject, visited, cloneObject, typeToReflect.BaseType, BindingFlags.Instance | BindingFlags.NonPublic, info => info.IsPrivate);
      }
    }

    private static void CopyFields(object originalObject, IDictionary<object, object> visited, object cloneObject, Type typeToReflect, BindingFlags bindingFlags = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.FlattenHierarchy, Func<FieldInfo, bool> filter = null)
    {
      foreach (FieldInfo fieldInfo in typeToReflect.GetFields(bindingFlags))
      {
        if (filter != null && filter(fieldInfo) == false) continue;
        if (IsPrimitive(fieldInfo.FieldType)) continue;
        var originalFieldValue = fieldInfo.GetValue(originalObject);
        var clonedFieldValue = InternalCopy(originalFieldValue, visited);
        fieldInfo.SetValue(cloneObject, clonedFieldValue);
      }
    }
    internal static T Copy<T>(T original)
    {
      return (T)Copy((Object)original);
    }


    class ReferenceEqualityComparer : EqualityComparer<Object>
    {
      public override bool Equals(object x, object y)
      {
        return ReferenceEquals(x, y);
      }
      public override int GetHashCode(object obj)
      {
        if (obj == null) return 0;
        return obj.GetHashCode();
      }
    }

    static class ArrayExtensions
    {
      internal static void ForEach(Array array, Action<Array, int[]> action)
      {
        if (array.LongLength == 0) return;
        ArrayTraverse walker = new ArrayTraverse(array);
        do action(array, walker.position);
        while (walker.Step());
      }
    }

    internal class ArrayTraverse
    {
      internal int[] position;
      private int[] maxLengths;

      public ArrayTraverse(Array array)
      {
        maxLengths = new int[array.Rank];
        for (int i = 0; i < array.Rank; ++i)
        {
          maxLengths[i] = array.GetLength(i) - 1;
        }
        position = new int[array.Rank];
      }

      internal bool Step()
      {
        for (int i = 0; i < position.Length; ++i)
        {
          if (position[i] < maxLengths[i])
          {
            position[i]++;
            for (int j = 0; j < i; j++)
            {
              position[j] = 0;
            }
            return true;
          }
        }
        return false;
      }
    }
  }
}
