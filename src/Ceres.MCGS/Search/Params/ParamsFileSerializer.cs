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
using System.Text.Json;
using System.Text.Json.Serialization;

#endregion

namespace Ceres.MCGS.Search.Params;

/// <summary>
/// Container holding a complete set of search parameters (the core ParamsSearch and the
/// leaf-selection ParamsSelect) so that both can be serialized together to a single JSON file.
/// </summary>
public record ParamsFileContents
{
  /// <summary>
  /// Core search algorithm parameters (includes the nested Execution object).
  /// </summary>
  public ParamsSearch ParamsSearch { get; set; }

  /// <summary>
  /// Leaf selection (PUCT) parameters.
  /// </summary>
  public ParamsSelect ParamsSelect { get; set; }
}


/// <summary>
/// Helper for serializing and deserializing complete sets of Ceres search parameters
/// (ParamsSearch and ParamsSelect, including the nested ParamsSearchExecution and other
/// nested objects) to and from JSON files.
///
/// These files can be generated via the "save-params" UCI command and reloaded via the
/// "load-params" UCI command, allowing a complete parameter configuration to be applied at
/// runtime without recompiling or issuing many individual setoption commands.
///
/// Note: the parameter objects expose their settings as public fields (not properties), so the
/// serializer is configured with IncludeFields = true. Enums are written by name for readability.
/// </summary>
public static class ParamsFileSerializer
{
  /// <summary>
  /// JSON options shared by Save and Load. Public fields are included (the parameter objects use
  /// fields, not properties), output is indented for human editing, and enums are written by name.
  /// </summary>
  static readonly JsonSerializerOptions jsonOptions = new JsonSerializerOptions()
  {
    WriteIndented = true,
    IncludeFields = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.Never,
    Converters = { new JsonStringEnumConverter() }
  };


  /// <summary>
  /// Serializes the specified search parameters to a JSON file at the given path.
  /// </summary>
  /// <param name="path">target file path</param>
  /// <param name="search">core search parameters (must be non-null)</param>
  /// <param name="select">leaf selection parameters (must be non-null)</param>
  public static void Save(string path, ParamsSearch search, ParamsSelect select)
  {
    ArgumentException.ThrowIfNullOrWhiteSpace(path);
    ArgumentNullException.ThrowIfNull(search);
    ArgumentNullException.ThrowIfNull(select);

    ParamsFileContents contents = new ParamsFileContents()
    {
      ParamsSearch = search,
      ParamsSelect = select
    };

    string json = JsonSerializer.Serialize(contents, jsonOptions);
    File.WriteAllText(path, json);
  }


  /// <summary>
  /// Reads and deserializes a set of search parameters previously written by Save.
  /// Throws a descriptive exception if the file is missing or cannot be parsed into both objects.
  /// </summary>
  /// <param name="path">source file path</param>
  /// <returns>the deserialized ParamsFileContents (both ParamsSearch and ParamsSelect non-null)</returns>
  public static ParamsFileContents Load(string path)
  {
    ArgumentException.ThrowIfNullOrWhiteSpace(path);

    if (!File.Exists(path))
    {
      throw new FileNotFoundException($"Params file not found: {path}", path);
    }

    string json = File.ReadAllText(path);

    ParamsFileContents contents;
    try
    {
      contents = JsonSerializer.Deserialize<ParamsFileContents>(json, jsonOptions);
    }
    catch (Exception exc)
    {
      throw new Exception($"Failed to parse params file {path}: {exc.Message}", exc);
    }

    if (contents?.ParamsSearch == null || contents.ParamsSelect == null)
    {
      throw new Exception($"Params file {path} did not contain both ParamsSearch and ParamsSelect objects.");
    }

    return contents;
  }
}
