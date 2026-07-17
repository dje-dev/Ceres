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
using System.Text;

#endregion

namespace Ceres.MCGS.Search.QProbeSelect;

/// <summary>
/// Immutable in-memory representation of a Q-uncertainty inline model (.qnn v3 blob,
/// written by qprobetrainpy exportinline.py). The model is a plain MLP mapping RAW
/// inputs (capture-time node/edge/parent features plus 2 dN-conditioning inputs
/// appended at the end) to a 2-output head: row 0 = mu (expected dQPure after dN
/// further visits), row 1 = log sigma. The frozen input normalization is pre-folded
/// into the first Linear and the target scale into the head, so consumers feed RAW
/// inputs and read dQ units directly; sigma = clamp(exp(out[1]), SIGMA_MIN, SIGMA_MAX).
/// All Linear weights are stored [out, in] row-major.
///
/// Instances are immutable after Load and safe for concurrent readers.
/// </summary>
public sealed class QProbeSelectNNModel
{
  public enum ActivationKind
  {
    SiLU = 0,
    ReLU = 1,
  }

  /// <summary>Variant id of the MLP uncertainty model (the only supported v3 variant).</summary>
  public const uint VARIANT_MLP_UNCERTAINTY = 2;

  /// <summary>Number of head outputs: (mu, log sigma).</summary>
  public const int NUM_OUTPUTS = 2;

  /// <summary>Defensive clamp applied to exp(log sigma) by consumers (mirrors exportinline.py).</summary>
  public const float SIGMA_MIN = 1e-6f;
  public const float SIGMA_MAX = 2.0f;

  public ActivationKind Activation { get; private set; }

  /// <summary>Number of RAW model inputs (capture features + 2 dN-conditioning inputs).</summary>
  public int NumInputs { get; private set; }

  public int NumHiddenLayers { get; private set; }
  public int HiddenDim { get; private set; }

  /// <summary>Reference horizons the model was harvested/trained around (dN is a free input).</summary>
  public int HorizonN1 { get; private set; }
  public int HorizonN2 { get; private set; }

  public ulong NNNetHash { get; private set; }
  public float RMSEps { get; private set; }
  public byte[] SchemaSHA1 { get; private set; }
  public string[] InputNames { get; private set; }
  public long NumParams { get; private set; }
  public string Path { get; private set; }

  // MLP blocks: per hidden layer i, Linear [out,in] + bias + RMSNorm weight.
  public float[][] MlpW;
  public float[][] MlpB;
  public float[][] MlpRms;

  // Head Linear [2, HiddenDim]: row 0 = mu (dQ units), row 1 = log sigma.
  public float[] HeadW, HeadB;


  /// <summary>
  /// Loads and validates a .qnn v3 blob from disk.
  /// </summary>
  public static QProbeSelectNNModel Load(string path)
  {
    using FileStream stream = File.OpenRead(path);
    using BinaryReader reader = new BinaryReader(stream);

    byte[] magic = reader.ReadBytes(8);
    if (Encoding.ASCII.GetString(magic) != "CMQPNN01")
    {
      throw new Exception($"QProbeSelectNNModel: bad magic in {path}");
    }

    uint version = reader.ReadUInt32();
    if (version != 3)
    {
      throw new Exception($"QProbeSelectNNModel: unsupported version {version} in {path} "
                        + "(version 3 mlp-uncertainty required; v2 ranking models are obsolete)");
    }

    QProbeSelectNNModel model = new QProbeSelectNNModel();
    model.Path = path;

    uint variant = reader.ReadUInt32();
    if (variant != VARIANT_MLP_UNCERTAINTY)
    {
      throw new Exception($"QProbeSelectNNModel: unsupported variant {variant} in {path}");
    }
    model.NumInputs = (int)reader.ReadUInt32();
    model.Activation = (ActivationKind)reader.ReadUInt32();
    model.NumHiddenLayers = (int)reader.ReadUInt32();
    model.HiddenDim = (int)reader.ReadUInt32();
    model.HorizonN1 = (int)reader.ReadUInt32();
    model.HorizonN2 = (int)reader.ReadUInt32();
    reader.ReadUInt32(); // reserved
    if (model.NumInputs < 4 || model.NumInputs > 512
     || model.NumHiddenLayers < 1 || model.NumHiddenLayers > 16
     || model.HiddenDim < 8 || model.HiddenDim > 4096)
    {
      throw new Exception($"QProbeSelectNNModel: implausible geometry in {path} "
                        + $"(inputs {model.NumInputs}, layers {model.NumHiddenLayers}, hidden {model.HiddenDim})");
    }
    if (model.HorizonN1 < 1 || model.HorizonN2 <= model.HorizonN1)
    {
      throw new Exception($"QProbeSelectNNModel: invalid horizons {model.HorizonN1}/{model.HorizonN2} in {path}");
    }
    model.NNNetHash = reader.ReadUInt64();
    model.RMSEps = reader.ReadSingle();
    model.SchemaSHA1 = reader.ReadBytes(20);

    uint numTensors = reader.ReadUInt32();
    Dictionary<string, (int[] Dims, float[] Data)> tensors = new Dictionary<string, (int[], float[])>((int)numTensors);
    long numParams = 0;
    for (uint t = 0; t < numTensors; t++)
    {
      uint nameLen = reader.ReadUInt32();
      string name = Encoding.ASCII.GetString(reader.ReadBytes((int)nameLen));
      uint ndim = reader.ReadUInt32();
      int[] dims = new int[ndim];
      long count = 1;
      for (uint d = 0; d < ndim; d++)
      {
        dims[d] = (int)reader.ReadUInt32();
        count *= dims[d];
      }
      float[] data = new float[count];
      byte[] raw = reader.ReadBytes((int)(count * sizeof(float)));
      Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
      tensors[name] = (dims, data);
      numParams += count;
    }
    model.NumParams = numParams;

    uint numInputNames = reader.ReadUInt32();
    model.InputNames = new string[numInputNames];
    for (uint f = 0; f < numInputNames; f++)
    {
      uint len = reader.ReadUInt32();
      model.InputNames[f] = Encoding.ASCII.GetString(reader.ReadBytes((int)len));
    }

    // NOTE: validation of the input-name order against a compiled live-extraction schema
    // is a search-integration concern (returns with Phase 2); here only internal
    // consistency of the blob is checked.
    if (model.InputNames.Length != model.NumInputs)
    {
      throw new Exception($"QProbeSelectNNModel: input-name trailer count {model.InputNames.Length} "
                        + $"!= numInputs {model.NumInputs} in {path}");
    }

    model.ResolveTensors(tensors);
    return model;
  }


  /// <summary>
  /// Resolves the named tensor dictionary into the typed weight fields,
  /// asserting the expected shapes.
  /// </summary>
  void ResolveTensors(Dictionary<string, (int[] Dims, float[] Data)> tensors)
  {
    MlpW = new float[NumHiddenLayers][];
    MlpB = new float[NumHiddenLayers][];
    MlpRms = new float[NumHiddenLayers][];
    for (int i = 0; i < NumHiddenLayers; i++)
    {
      int inDim = i == 0 ? NumInputs : HiddenDim;
      MlpW[i] = Take(tensors, $"mlp.{i}.w", HiddenDim, inDim);
      MlpB[i] = Take(tensors, $"mlp.{i}.b", HiddenDim);
      MlpRms[i] = Take(tensors, $"mlp.{i}.rms", HiddenDim);
    }
    HeadW = Take(tensors, "head.w", NUM_OUTPUTS, HiddenDim);
    HeadB = Take(tensors, "head.b", NUM_OUTPUTS);
  }


  /// <summary>
  /// Fetches a named tensor, asserting its shape matches expectedDims.
  /// </summary>
  static float[] Take(Dictionary<string, (int[] Dims, float[] Data)> tensors,
                      string name, params int[] expectedDims)
  {
    if (!tensors.TryGetValue(name, out (int[] Dims, float[] Data) entry))
    {
      throw new Exception($"QProbeSelectNNModel: missing tensor {name}");
    }
    if (entry.Dims.Length != expectedDims.Length)
    {
      throw new Exception($"QProbeSelectNNModel: tensor {name} rank {entry.Dims.Length} != {expectedDims.Length}");
    }
    for (int i = 0; i < expectedDims.Length; i++)
    {
      if (entry.Dims[i] != expectedDims[i])
      {
        throw new Exception($"QProbeSelectNNModel: tensor {name} dim[{i}]={entry.Dims[i]} != {expectedDims[i]}");
      }
    }
    return entry.Data;
  }


  /// <summary>
  /// Returns short descriptive string.
  /// </summary>
  public override string ToString()
    => $"<QProbeSelectNNModel mlp-uncertainty {NumInputs}->{NumHiddenLayers}x{HiddenDim}->2 "
     + $"act={Activation} horizons={HorizonN1}/{HorizonN2} params={NumParams:N0} {Path}>";
}
