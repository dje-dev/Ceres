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
using System.Buffers.Binary;
using System.IO;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;

#endregion

namespace Ceres.Chess.NNEvaluators.Remote
{
  /// <summary>
  /// Message types exchanged between NNRemote client and server.
  /// </summary>
  public enum NNRemoteMessageType : uint
  {
    // Client -> Server
    Handshake = 1,
    EvalRequest = 2,
    Disconnect = 3,
    Ping = 4,

    // Server -> Client
    HandshakeAck = 101,
    EvalResponse = 102,
    Error = 103,
    Pong = 104,
  }


  /// <summary>
  /// Exception thrown when a remote NN evaluation server returns an error.
  /// </summary>
  public class NNRemoteEvaluationException : Exception
  {
    public readonly string RemoteExceptionType;

    public NNRemoteEvaluationException(string remoteExceptionType, string remoteMessage)
      : base($"Remote evaluation error ({remoteExceptionType}): {remoteMessage}")
    {
      RemoteExceptionType = remoteExceptionType;
    }
  }


  /// <summary>
  /// Exception thrown when the TCP connection to the remote server fails.
  /// </summary>
  public class NNRemoteConnectionException : Exception
  {
    public NNRemoteConnectionException(string message, Exception innerException = null)
      : base(message, innerException) { }
  }


  /// <summary>
  /// Wire protocol constants and message framing helpers for remote NN evaluation.
  ///
  /// Message format:
  ///   [4 bytes] MessageType (uint32, little-endian)
  ///   [4 bytes] PayloadLength (uint32, uncompressed size)
  ///   [4 bytes] CompressedLength (uint32, 0 if uncompressed)
  ///   [PayloadLength or CompressedLength bytes] Payload
  /// </summary>
  public static class NNRemoteProtocol
  {
    /// <summary>
    /// Size of the message header in bytes.
    /// </summary>
    public const int HEADER_SIZE = 12;

    /// <summary>
    /// Default TCP port for the remote evaluation server.
    /// </summary>
    public const int DEFAULT_PORT = 50055;

    /// <summary>
    /// Protocol version for handshake compatibility checking.
    /// </summary>
    public const int PROTOCOL_VERSION = 1;

    /// <summary>
    /// Minimum payload size to attempt compression (bytes).
    /// Below this threshold, compression overhead exceeds benefit.
    /// </summary>
    public const int COMPRESSION_THRESHOLD = 4096;

    /// <summary>
    /// Zstd compression level (1 = fastest).
    /// </summary>
    public const int ZSTD_LEVEL = 1;

    /// <summary>
    /// Default read timeout in milliseconds (30 minutes, to allow for
    /// slow evaluator initialization such as TensorRT engine compilation).
    /// </summary>
    public const int DEFAULT_READ_TIMEOUT_MS = 30 * 60 * 1000;


    /// <summary>
    /// Writes a framed message to the network stream.
    /// </summary>
    public static void WriteMessage(NetworkStream stream, NNRemoteMessageType type,
                                     ReadOnlySpan<byte> payload, bool useCompression,
                                     byte[] compressBuffer)
    {
      Span<byte> header = stackalloc byte[HEADER_SIZE];
      int payloadLength = payload.Length;
      int compressedLength = 0;
      ReadOnlySpan<byte> dataToSend = payload;

      if (useCompression && payloadLength > COMPRESSION_THRESHOLD)
      {
        int maxCompressedSize = ZstdBlockCompress.CompressBound(payloadLength);
        if (compressBuffer == null || compressBuffer.Length < maxCompressedSize)
        {
          compressBuffer = new byte[maxCompressedSize];
        }

        int actualCompressed = ZstdBlockCompress.Compress(payload, compressBuffer, ZSTD_LEVEL);

        // Only use compression if it actually reduced size.
        if (actualCompressed < payloadLength)
        {
          compressedLength = actualCompressed;
          dataToSend = compressBuffer.AsSpan(0, actualCompressed);
        }
      }

      BinaryPrimitives.WriteUInt32LittleEndian(header, (uint)type);
      BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(4), (uint)payloadLength);
      BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(8), (uint)compressedLength);

      stream.Write(header);
      if (dataToSend.Length > 0)
      {
        stream.Write(dataToSend);
      }
      stream.Flush();
    }


    /// <summary>
    /// Reads a framed message from the network stream.
    /// Returns the message type and the decompressed payload in the provided buffer.
    /// The buffer is resized if necessary.
    /// </summary>
    public static (NNRemoteMessageType type, int payloadLength) ReadMessage(
      NetworkStream stream, ref byte[] buffer, ref byte[] decompressBuffer)
    {
      Span<byte> header = stackalloc byte[HEADER_SIZE];
      ReadExact(stream, header);

      NNRemoteMessageType type = (NNRemoteMessageType)BinaryPrimitives.ReadUInt32LittleEndian(header);
      int payloadLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(4));
      int compressedLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(8));

      if (payloadLength == 0)
      {
        return (type, 0);
      }

      if (compressedLength > 0)
      {
        // Read compressed data into decompressBuffer, then decompress into buffer.
        EnsureBufferSize(ref decompressBuffer, compressedLength);
        ReadExact(stream, decompressBuffer.AsSpan(0, compressedLength));

        EnsureBufferSize(ref buffer, payloadLength);
        int decompressed = ZstdBlockCompress.Decompress(
          decompressBuffer.AsSpan(0, compressedLength),
          buffer.AsSpan(0, payloadLength));

        if (decompressed != payloadLength)
        {
          throw new IOException($"Zstd decompression size mismatch: expected {payloadLength}, got {decompressed}");
        }
      }
      else
      {
        // Uncompressed: read directly into buffer.
        EnsureBufferSize(ref buffer, payloadLength);
        ReadExact(stream, buffer.AsSpan(0, payloadLength));
      }

      return (type, payloadLength);
    }


    /// <summary>
    /// Writes a length-prefixed UTF-8 string to a span, returns bytes written.
    /// </summary>
    public static int WriteString(Span<byte> dest, string value)
    {
      if (value == null)
      {
        BinaryPrimitives.WriteInt32LittleEndian(dest, -1);
        return 4;
      }

      int byteCount = Encoding.UTF8.GetByteCount(value);
      BinaryPrimitives.WriteInt32LittleEndian(dest, byteCount);
      Encoding.UTF8.GetBytes(value.AsSpan(), dest.Slice(4));
      return 4 + byteCount;
    }


    /// <summary>
    /// Reads a length-prefixed UTF-8 string from a span, returns (value, bytesConsumed).
    /// </summary>
    public static (string value, int bytesConsumed) ReadString(ReadOnlySpan<byte> src)
    {
      int byteCount = BinaryPrimitives.ReadInt32LittleEndian(src);
      if (byteCount == -1)
      {
        return (null, 4);
      }

      string value = Encoding.UTF8.GetString(src.Slice(4, byteCount));
      return (value, 4 + byteCount);
    }


    /// <summary>
    /// Reads exactly count bytes from the stream, blocking until complete.
    /// </summary>
    static void ReadExact(NetworkStream stream, Span<byte> buffer)
    {
      int totalRead = 0;
      while (totalRead < buffer.Length)
      {
        int read = stream.Read(buffer.Slice(totalRead));
        if (read == 0)
        {
          throw new IOException("Connection closed by remote host.");
        }
        totalRead += read;
      }
    }


    /// <summary>
    /// Ensures the buffer is at least the specified size.
    /// </summary>
    static void EnsureBufferSize(ref byte[] buffer, int requiredSize)
    {
      if (buffer == null || buffer.Length < requiredSize)
      {
        buffer = new byte[requiredSize];
      }
    }
  }


  /// <summary>
  /// Block-mode Zstd compression/decompression via P/Invoke.
  /// Uses the single-shot ZSTD_compress/ZSTD_decompress functions
  /// for lower overhead than the streaming API.
  /// </summary>
  internal static class ZstdBlockCompress
  {
    [DllImport("libzstd", CallingConvention = CallingConvention.Cdecl)]
    static extern UIntPtr ZSTD_compress(IntPtr dst, UIntPtr dstCapacity,
                                         IntPtr src, UIntPtr srcSize,
                                         int compressionLevel);

    [DllImport("libzstd", CallingConvention = CallingConvention.Cdecl)]
    static extern UIntPtr ZSTD_decompress(IntPtr dst, UIntPtr dstCapacity,
                                           IntPtr src, UIntPtr srcSize);

    [DllImport("libzstd", CallingConvention = CallingConvention.Cdecl)]
    static extern UIntPtr ZSTD_compressBound(UIntPtr srcSize);

    [DllImport("libzstd", CallingConvention = CallingConvention.Cdecl)]
    static extern uint ZSTD_isError(UIntPtr code);

    [DllImport("libzstd", CallingConvention = CallingConvention.Cdecl)]
    static extern IntPtr ZSTD_getErrorName(UIntPtr code);


    public static int CompressBound(int srcSize)
    {
      return (int)ZSTD_compressBound((UIntPtr)srcSize);
    }


    public static unsafe int Compress(ReadOnlySpan<byte> src, Span<byte> dst, int level)
    {
      fixed (byte* pSrc = src)
      fixed (byte* pDst = dst)
      {
        UIntPtr result = ZSTD_compress((IntPtr)pDst, (UIntPtr)dst.Length,
                                        (IntPtr)pSrc, (UIntPtr)src.Length, level);
        ThrowIfError(result);
        return (int)result;
      }
    }


    public static unsafe int Decompress(ReadOnlySpan<byte> src, Span<byte> dst)
    {
      fixed (byte* pSrc = src)
      fixed (byte* pDst = dst)
      {
        UIntPtr result = ZSTD_decompress((IntPtr)pDst, (UIntPtr)dst.Length,
                                          (IntPtr)pSrc, (UIntPtr)src.Length);
        ThrowIfError(result);
        return (int)result;
      }
    }


    static void ThrowIfError(UIntPtr code)
    {
      if (ZSTD_isError(code) != 0)
      {
        string msg = Marshal.PtrToStringAnsi(ZSTD_getErrorName(code));
        throw new IOException($"Zstd error: {msg}");
      }
    }
  }
}
