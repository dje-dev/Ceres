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
using System.Security.Policy;
using System.Text;

#endregion

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// Parser of strings consisting of space separated
  /// sequences of key=value pairs.
  /// </summary>
  public class KeyValueSetParsed
  {
    /// <summary>
    /// The parsed set of key/value pairs.
    /// </summary>
    public readonly List<(string Key, string Value)> KeyValuePairs;


    /// <summary>
    /// Constructor from a specification string to be parsed.
    /// </summary>
    /// <param name="args"></param>
    public KeyValueSetParsed(string args, IList<string> validKeys)
    {
      KeyValuePairs = new List<(string, string)>();
      if (args != null && args != "")
      {
        string[] parts = args.Replace("  ", " ").Split(" ");

        foreach (string part in parts)
        {
          string[] kvpParts = part.Split("=");
          if (kvpParts.Length == 2)
            KeyValuePairs.Add((kvpParts[0], kvpParts[1]));
          else
            throw new Exception("Expected key=value pair (without spaces)");
        }
      }

      if (validKeys != null && validKeys.Count > 0) ValidateKeys(validKeys);
    }


    /// <summary>
    /// Returns the value associated with a key of specified name, 
    /// possibly with a supplied default value and validation method.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="key"></param>
    /// <param name="defaultValue"></param>
    /// <param name="defaultMustExist"></param>
    /// <param name="converterFunc"></param>
    /// <returns></returns>
    public T GetValueOrDefaultMapped<T>(string key, string defaultValue, bool defaultMustExist, Func<string, T> converterFunc)
    {
      string strValue = GetValueOrDefault(key, defaultValue, defaultMustExist);

      if (strValue == null)
        return default(T);
      else
        return converterFunc(strValue);
    }


    /// <summary>
    /// Returns the value associated with a key of specified name, 
    /// possibly with a supplied default value.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="defaultValue"></param>
    /// <param name="defaultMustExist"></param>
    /// <returns></returns>
    public string GetValueOrDefault(string key, string defaultValue, bool defaultMustExist)
    {
      string value = GetValue(key);
      if (value == null)
      {
        if (defaultValue != null)
          return defaultValue;
        else if (defaultMustExist)
          throw new Exception($"Error: {key} required but not present nor in user settings");
        else
          return null;
      }
      else
        return value;
    }


    /// <summary>
    /// Returns the value associated with a key of specified name,
    /// throwing Exception with specified text if the key was not found.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="errorMessage"></param>
    /// <returns></returns>
    public string GetRequiredValue(string key, string errorMessage)
    {
      string value = GetValue(key);
      if (value == null)
        throw new Exception($"Required key  {key} not found. {errorMessage}");
      else
        return value;
    }


    /// <summary>
    /// Returns the value associated with a key of specified name.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public string GetValue(string key)
    {
      foreach (var kvp in KeyValuePairs)
        if (kvp.Item1.ToLower() == key.ToLower())
          return kvp.Item2;

      return null;
    }

    #region Key validation

    string StringList(IList<string> values)
    {
      string validKeysStr = "";
      int count = 0;
      foreach (string value in values)
        validKeysStr += (count++ == 0 ? "" : ", ") + value;

      return validKeysStr;
    }

    /// <summary>
    /// Validates that all Keys are contained within a specified set of valid keys.
    /// </summary>
    /// <param name="validKeys"></param>
    void ValidateKeys(IList<string> validKeys)
    {
      foreach ((string Key, string Value) in KeyValuePairs)
        if (!validKeys.Contains(Key.ToUpper()))
          throw new Exception($"Specified key {Key} not in valid values of {StringList(validKeys)}");
    }

    #endregion

  }
}
