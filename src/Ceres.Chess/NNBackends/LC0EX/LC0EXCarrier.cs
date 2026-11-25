#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Provenance

/*
  lc0ex -- the executable format, its precompiled kernels, the builder that produces them and the
  reference CUDA runtime -- is the work of Kovax (GitHub: Kovax007) for Leela Chess Zero.

  This file is a C# port of parts of that work and of Leela Chess Zero itself:
    src/neural/backends/lc0ex-cuda/network_lc0ex_cuda.cc     (reading the carrier's ONNX initializers)
    src/neural/backends/lc0ex-cuda/network_fingerprint.cc    (the network architecture fingerprint)
    src/neural/loader.cc, FixOlderWeightsFile                (upgrading older network format flags)

  Original sources: https://github.com/Kovax007/lc0, branch tcec-season-30-swiss-lc0ex, commit 29ec4b44.
  Copyright (C) 2026 The LCZero Authors. Licensed under the GNU General Public License, version 3 or
  (at your option) any later version, with the additional permission for NVIDIA libraries stated there.

  Ported to C# and modified for Ceres in November 2026.
  Differences from the original: the file is walked at the protobuf wire level so the weights are
  not copied on the host.
*/

#endregion

#region Using directives

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

#endregion

namespace Ceres.Chess.NNBackends.LC0EX
{
  /// <summary>
  /// The subset of Lc0 NetworkFormat fields (net.proto) relevant to lc0ex,
  /// with the smolgen/FFN activations resolved against the default activation
  /// exactly as in lc0's lc0ex network fingerprint (network_fingerprint.cc).
  /// </summary>
  public readonly record struct LC0EXNetworkFormat(int Input, int Output, int Network, int Policy, int Value,
                                                   int MovesLeft, int DefaultActivation, int SmolgenActivation,
                                                   int FFNActivation, int InputEmbedding)
  {
    public const int INPUT_CLASSICAL_112_PLANE = 1;
    public const int OUTPUT_WDL = 2;
    public const int MOVES_LEFT_V1 = 1;
  }


  /// <summary>
  /// The architecture fingerprint lc0 uses to decide if an lc0ex executable matches a network
  /// (see lc0 BuildNetworkFingerprint): the resolved network format, the attention head count
  /// and the number of encoder layers of the network's weights section (both zero for
  /// a carrier holding only an ONNX model).
  /// </summary>
  public readonly record struct LC0EXNetworkFingerprint(LC0EXNetworkFormat Format, int Headcount, int NumEncoderLayers)
  {
    // net.proto field numbers and enum values used below.
    const int NET_FORMAT = 4;
    const int NET_WEIGHTS = 10;
    const int FORMAT_NETWORK_FORMAT = 2;
    const int WEIGHTS_ENCODER = 27;
    const int WEIGHTS_HEADCOUNT = 28;
    const int WEIGHTS_SMOLGEN_W = 35;
    const int WEIGHTS_VALUE_HEADS = 44;
    const int WEIGHTS_POLICY_HEADS = 45;

    const int NF_INPUT = 1, NF_OUTPUT = 2, NF_NETWORK = 3, NF_POLICY = 4, NF_VALUE = 5, NF_MOVES_LEFT = 6,
              NF_DEFAULT_ACTIVATION = 7, NF_SMOLGEN_ACTIVATION = 8, NF_FFN_ACTIVATION = 9, NF_INPUT_EMBEDDING = 10;

    const int OUTPUT_CLASSICAL = 1;
    const int NETWORK_CLASSICAL = 1;
    const int NETWORK_SE = 2;
    const int NETWORK_CLASSICAL_WITH_HEADFORMAT = 3;
    const int NETWORK_SE_WITH_HEADFORMAT = 4;
    const int NETWORK_ATTENTIONBODY_WITH_HEADFORMAT = 6;
    const int NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT = 7;
    const int NETWORK_AB_LEGACY_WITH_MULTIHEADFORMAT = 134;
    const int POLICY_CLASSICAL = 1;
    const int VALUE_CLASSICAL = 1;
    const int DEFAULT_ACTIVATION_RELU = 0;
    const int DEFAULT_ACTIVATION_MISH = 1;
    const int ACTIVATION_DEFAULT = 0;
    const int ACTIVATION_MISH = 1;
    const int ACTIVATION_RELU = 2;
    const int ACTIVATION_SWISH = 7;
    const int ACTIVATION_RELU_2 = 8;
    const int INPUT_EMBEDDING_PE_MAP = 1;
    const int INPUT_EMBEDDING_PE_DENSE = 2;


    /// <summary>
    /// Computes the fingerprint of a serialized pblczero.Net.
    /// </summary>
    /// <param name="applyLoaderFixups">
    /// true for a network file (carrier): the format is first upgraded as lc0 does on load
    /// (FixOlderWeightsFile in lc0 loader.cc), so files written with older format flags compare
    /// equal to what lc0 would compute. false for an executable's metadata, which already is a fingerprint.
    /// </param>
    /// <param name="hasNetworkFormat">if the Net had a network format at all</param>
    /// <param name="hasWeightsSection">if the Net had an Lc0 weights section (the source of head and layer counts)</param>
    public static LC0EXNetworkFingerprint FromSerializedNet(byte[] data, int start, int length, bool applyLoaderFixups,
                                                            out bool hasNetworkFormat, out bool hasWeightsSection)
    {
      int[] f = new int[NF_INPUT_EMBEDDING + 1];
      bool[] has = new bool[NF_INPUT_EMBEDDING + 1];
      hasNetworkFormat = false;
      hasWeightsSection = false;
      int headcount = 0;
      int numEncoders = 0;
      bool hasSmolgenW = false, hasValueHeads = false, hasPolicyHeads = false;

      LC0EXProtoReader net = new(data, start, length);
      while (net.Next(out int field, out int wt))
      {
        if (field == NET_FORMAT && wt == LC0EXProtoReader.WIRE_LENGTH_DELIMITED)
        {
          LC0EXProtoReader format = net.ReadMessage();
          while (format.Next(out int ff, out int fwt))
          {
            if (ff == FORMAT_NETWORK_FORMAT && fwt == LC0EXProtoReader.WIRE_LENGTH_DELIMITED)
            {
              hasNetworkFormat = true;
              LC0EXProtoReader nf = format.ReadMessage();
              while (nf.Next(out int nff, out int nfwt))
              {
                if (nff >= 1 && nff <= NF_INPUT_EMBEDDING && nfwt == LC0EXProtoReader.WIRE_VARINT)
                {
                  f[nff] = (int)nf.ReadVarint();
                  has[nff] = true;
                }
                else
                {
                  nf.Skip(nfwt);
                }
              }
            }
            else
            {
              format.Skip(fwt);
            }
          }
        }
        else if (field == NET_WEIGHTS && wt == LC0EXProtoReader.WIRE_LENGTH_DELIMITED)
        {
          // Only the top level of the weights is walked (the layer payloads are skipped, not read).
          hasWeightsSection = true;
          LC0EXProtoReader weights = net.ReadMessage();
          while (weights.Next(out int wf, out int wwt))
          {
            switch (wf)
            {
              case WEIGHTS_ENCODER: numEncoders++; break;
              case WEIGHTS_SMOLGEN_W: hasSmolgenW = true; break;
              case WEIGHTS_VALUE_HEADS: hasValueHeads = true; break;
              case WEIGHTS_POLICY_HEADS: hasPolicyHeads = true; break;
            }

            if (wf == WEIGHTS_HEADCOUNT && wwt == LC0EXProtoReader.WIRE_VARINT)
            {
              headcount = (int)weights.ReadVarint();
            }
            else
            {
              weights.Skip(wwt);
            }
          }
        }
        else
        {
          net.Skip(wt);
        }
      }

      if (applyLoaderFixups)
      {
        // Port of lc0 FixOlderWeightsFile (src/neural/loader.cc).
        int network = f[NF_NETWORK];
        if (!hasNetworkFormat)
        {
          f[NF_INPUT] = LC0EXNetworkFormat.INPUT_CLASSICAL_112_PLANE;
          f[NF_OUTPUT] = OUTPUT_CLASSICAL;
          f[NF_NETWORK] = NETWORK_CLASSICAL_WITH_HEADFORMAT;
          f[NF_VALUE] = VALUE_CLASSICAL;
          f[NF_POLICY] = POLICY_CLASSICAL;
        }
        else if (network == NETWORK_CLASSICAL)
        {
          f[NF_NETWORK] = NETWORK_CLASSICAL_WITH_HEADFORMAT;
          f[NF_VALUE] = VALUE_CLASSICAL;
          f[NF_POLICY] = POLICY_CLASSICAL;
        }
        else if (network == NETWORK_SE)
        {
          f[NF_NETWORK] = NETWORK_SE_WITH_HEADFORMAT;
          f[NF_VALUE] = VALUE_CLASSICAL;
          f[NF_POLICY] = POLICY_CLASSICAL;
        }
        else if (network == NETWORK_SE_WITH_HEADFORMAT && numEncoders > 0)
        {
          // Attention body network made with an old protobuf.
          f[NF_NETWORK] = NETWORK_ATTENTIONBODY_WITH_HEADFORMAT;
          if (hasSmolgenW)
          {
            f[NF_FFN_ACTIVATION] = ACTIVATION_RELU_2;
            f[NF_SMOLGEN_ACTIVATION] = ACTIVATION_SWISH;
          }
        }
        else if (network == NETWORK_AB_LEGACY_WITH_MULTIHEADFORMAT)
        {
          f[NF_NETWORK] = NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT;
        }

        if (f[NF_NETWORK] == NETWORK_ATTENTIONBODY_WITH_HEADFORMAT)
        {
          if (hasPolicyHeads && hasValueHeads)
          {
            f[NF_NETWORK] = NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT;
            f[NF_INPUT_EMBEDDING] = INPUT_EMBEDDING_PE_DENSE;
            has[NF_INPUT_EMBEDDING] = true;
          }
          if (!has[NF_INPUT_EMBEDDING])
          {
            f[NF_INPUT_EMBEDDING] = INPUT_EMBEDDING_PE_MAP;
          }
        }
      }

      // Resolve the "default" activations as lc0's fingerprint does.
      int resolvedDefault = f[NF_DEFAULT_ACTIVATION] switch
      {
        DEFAULT_ACTIVATION_RELU => ACTIVATION_RELU,
        DEFAULT_ACTIVATION_MISH => ACTIVATION_MISH,
        _ => throw new NotSupportedException($"Unsupported default activation {f[NF_DEFAULT_ACTIVATION]} in network format.")
      };
      int smolgen = f[NF_SMOLGEN_ACTIVATION] == ACTIVATION_DEFAULT ? resolvedDefault : f[NF_SMOLGEN_ACTIVATION];
      int ffn = f[NF_FFN_ACTIVATION] == ACTIVATION_DEFAULT ? resolvedDefault : f[NF_FFN_ACTIVATION];

      LC0EXNetworkFormat networkFormat = new(f[NF_INPUT], f[NF_OUTPUT], f[NF_NETWORK], f[NF_POLICY], f[NF_VALUE],
                                             f[NF_MOVES_LEFT], f[NF_DEFAULT_ACTIVATION], smolgen, ffn, f[NF_INPUT_EMBEDDING]);
      return new LC0EXNetworkFingerprint(networkFormat, headcount, numEncoders);
    }
  }


  /// <summary>
  /// One raw-data initializer of the carrier's ONNX graph.
  /// The bytes are a zero-copy slice (Offset, Length) of LC0EXCarrier.Data.
  /// </summary>
  public readonly record struct LC0EXInitializer(string Name, int DataType, long[] Dims, int Offset, int Length);


  /// <summary>
  /// The weights for an lc0ex executable: an Lc0 .pb.gz "carrier" (written by the lc0ex builder's
  /// carrier tools) that holds a network format and an ONNX model whose initializers are named
  /// exactly like the executable's persistent buffers. A carrier may also still hold the classic
  /// Lc0 "weights" section of the net it was made from (as the BT4 carriers do); that section
  /// only contributes to the architecture fingerprint and is otherwise skipped.
  ///
  /// The file is decompressed once into a single array and walked at the wire level,
  /// so the several hundred MB of weights are never copied again on the host.
  /// </summary>
  public sealed class LC0EXCarrier
  {
    public string FileName { get; }

    /// <summary>
    /// The decompressed carrier (initializer raw data slices refer into this).
    /// </summary>
    public byte[] Data { get; }

    /// <summary>
    /// The architecture fingerprint of this network, as lc0 computes it.
    /// </summary>
    public LC0EXNetworkFingerprint Fingerprint { get; }

    public LC0EXNetworkFormat NetworkFormat => Fingerprint.Format;

    /// <summary>
    /// If the carrier still holds the Lc0 weights section of the net it was made from.
    /// Without it the head count and layer count of the fingerprint are unknown (zero).
    /// </summary>
    public bool HasWeightsSection { get; }

    public IReadOnlyDictionary<string, LC0EXInitializer> Initializers => initializers;

    readonly Dictionary<string, LC0EXInitializer> initializers = new();


    public LC0EXCarrier(string fileName)
    {
      if (!File.Exists(fileName))
      {
        throw new FileNotFoundException($"lc0ex carrier (weights) file not found: {fileName}");
      }

      FileName = fileName;
      Data = ReadPossiblyCompressed(fileName);

      bool haveOnnx = false;
      LC0EXProtoReader net = new(Data);
      while (net.Next(out int field, out int wt))
      {
        const int NET_ONNX_MODEL = 11;
        if (field == NET_ONNX_MODEL && wt == LC0EXProtoReader.WIRE_LENGTH_DELIMITED)
        {
          haveOnnx = true;
          ParseOnnxModel(net.ReadMessage());
        }
        else
        {
          net.Skip(wt);
        }
      }

      if (!haveOnnx)
      {
        // (lc0 would convert a plain weights file to ONNX in-process; that converter is not available here.)
        throw new NotSupportedException($"{fileName} contains no onnx_model. lc0ex in Ceres requires the lc0ex builder's "
                                      + "carrier file, in which the weights are ONNX initializers named like the executable's buffers.");
      }

      Fingerprint = LC0EXNetworkFingerprint.FromSerializedNet(Data, 0, Data.Length, applyLoaderFixups: true,
                                                              out bool hasNetworkFormat, out bool hasWeightsSection);
      HasWeightsSection = hasWeightsSection;
      if (!hasNetworkFormat)
      {
        throw new FormatException($"{fileName} contains no network format.");
      }

      if (NetworkFormat.Input != LC0EXNetworkFormat.INPUT_CLASSICAL_112_PLANE)
      {
        throw new NotSupportedException($"lc0ex carrier input format {NetworkFormat.Input} not supported; "
                                      + "Ceres supports only INPUT_CLASSICAL_112_PLANE (1).");
      }
      if (NetworkFormat.Output != LC0EXNetworkFormat.OUTPUT_WDL)
      {
        throw new NotSupportedException($"lc0ex carrier output format {NetworkFormat.Output} not supported; WDL (2) required.");
      }
    }


    /// <summary>
    /// OnnxModel { bytes model = 1; ... } -> ModelProto { GraphProto graph = 7; }
    ///   -> GraphProto { repeated TensorProto initializer = 5; }
    /// </summary>
    void ParseOnnxModel(LC0EXProtoReader onnxModel)
    {
      while (onnxModel.Next(out int field, out int wt))
      {
        if (field != 1 || wt != LC0EXProtoReader.WIRE_LENGTH_DELIMITED)
        {
          onnxModel.Skip(wt);
          continue;
        }

        LC0EXProtoReader model = onnxModel.ReadMessage();
        while (model.Next(out int mf, out int mwt))
        {
          if (mf != 7 || mwt != LC0EXProtoReader.WIRE_LENGTH_DELIMITED)
          {
            model.Skip(mwt);
            continue;
          }

          LC0EXProtoReader graph = model.ReadMessage();
          while (graph.Next(out int gf, out int gwt))
          {
            if (gf == 5 && gwt == LC0EXProtoReader.WIRE_LENGTH_DELIMITED)
            {
              LC0EXInitializer init = ParseTensor(graph.ReadMessage());
              if (!initializers.TryAdd(init.Name, init))
              {
                throw new FormatException($"Duplicate initializer '{init.Name}' in lc0ex carrier {FileName}.");
              }
            }
            else
            {
              graph.Skip(gwt);
            }
          }
        }
      }
    }


    /// <summary>
    /// TensorProto { repeated int64 dims = 1; int32 data_type = 2; string name = 8;
    ///               bytes raw_data = 9; DataLocation data_location = 14; (typed data fields 4,5,6,7,10,11) }
    /// </summary>
    LC0EXInitializer ParseTensor(LC0EXProtoReader t)
    {
      List<ulong> dims = new();
      int dataType = 0;
      string name = null;
      int rawOffset = -1;
      int rawLength = 0;
      bool hasTypedData = false;
      int dataLocation = 0;

      while (t.Next(out int field, out int wt))
      {
        switch (field)
        {
          case 1: t.ReadRepeatedVarint(wt, dims); break;
          case 2: dataType = (int)t.ReadVarint(); break;
          case 8: name = t.ReadString(); break;
          case 9: (rawOffset, rawLength) = t.ReadBytesSlice(); break;
          case 14: dataLocation = (int)t.ReadVarint(); break;
          case 4:
          case 5:
          case 6:
          case 7:
          case 10:
          case 11:
            hasTypedData = true;
            t.Skip(wt);
            break;
          default: t.Skip(wt); break;
        }
      }

      if (name == null)
      {
        throw new FormatException($"Unnamed initializer in lc0ex carrier {FileName}.");
      }
      if (dataLocation != 0)
      {
        throw new NotSupportedException($"Initializer '{name}' uses external data, not supported for lc0ex carriers.");
      }
      if (rawOffset < 0)
      {
        // No raw_data. A tensor with no data at all is an empty tensor (as in lc0); one holding
        // typed data fields is unusable, which is only an error if a buffer actually needs it.
        rawLength = 0;
        if (!hasTypedData)
        {
          rawOffset = 0;
        }
      }

      long[] dimsArray = new long[dims.Count];
      for (int i = 0; i < dims.Count; i++)
      {
        dimsArray[i] = unchecked((long)dims[i]);
      }

      return new LC0EXInitializer(name, dataType, dimsArray, rawOffset, rawLength);
    }


    /// <summary>
    /// Reads a file, decompressing it if gzipped, into a single exactly-sized array
    /// (preallocated from the gzip ISIZE trailer to avoid MemoryStream doubling).
    /// </summary>
    static byte[] ReadPossiblyCompressed(string fileName)
    {
      using FileStream fs = new(fileName, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
      byte[] header = new byte[2];
      if (fs.Read(header, 0, 2) < 2 || header[0] != 0x1f || header[1] != 0x8b)
      {
        if (fs.Length > Array.MaxLength)
        {
          throw new NotSupportedException($"{fileName} is larger than the {Array.MaxLength} bytes supported for an lc0ex carrier.");
        }

        fs.Position = 0;
        byte[] raw = new byte[fs.Length];
        fs.ReadExactly(raw);
        return raw;
      }

      // ISIZE = uncompressed size mod 2^32 (last 4 bytes, little endian).
      long expected = 0;
      if (fs.Length >= 4)
      {
        fs.Position = fs.Length - 4;
        byte[] trailer = new byte[4];
        fs.ReadExactly(trailer);
        expected = BitConverter.ToUInt32(trailer, 0);
      }
      fs.Position = 0;

      byte[] buffer = new byte[Math.Clamp(expected, 1, Array.MaxLength)];
      int used = 0;
      using GZipStream gz = new(fs, CompressionMode.Decompress);
      while (true)
      {
        if (used == buffer.Length)
        {
          // Full. With a correct trailer this is the end of the data, so check for
          // that before growing (growing would copy the whole buffer twice for nothing).
          int next = gz.ReadByte();
          if (next < 0)
          {
            break;
          }

          // Trailer wrong (e.g. multi-member file, or 4 GB or more of data); grow.
          if (buffer.Length == Array.MaxLength)
          {
            throw new NotSupportedException($"{fileName} decompresses to more than the {Array.MaxLength} bytes "
                                          + "supported for an lc0ex carrier.");
          }
          Array.Resize(ref buffer, (int)Math.Min((long)buffer.Length * 2, Array.MaxLength));
          buffer[used++] = (byte)next;
          continue;
        }

        int n = gz.Read(buffer, used, buffer.Length - used);
        if (n == 0)
        {
          break;
        }
        used += n;
      }

      if (used != buffer.Length)
      {
        Array.Resize(ref buffer, used);
      }
      return buffer;
    }


    public override string ToString()
      => $"<LC0EXCarrier {Path.GetFileName(FileName)} initializers={initializers.Count} bytes={Data.Length / (1024 * 1024)}MB>";
  }
}
