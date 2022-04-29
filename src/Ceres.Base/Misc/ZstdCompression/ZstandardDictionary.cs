
// Source file taken from bp74/ZStandard.Net project on Github (under BSD license):
//   https://github.com/bp74/Zstandard.Net

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Interop = Zstandard.Net.ZstandardInterop;

namespace Zstandard.Net
{
  /// <summary>
  /// A Zstandard dictionary improves the compression ratio and speed on small data dramatically.
  /// </summary>
  /// <remarks>
  /// A Zstandard dictionary is calculated with a high number of small sample data.
  /// Please refer to the Zstandard documentation for more details.
  /// </remarks>
  /// <seealso cref="System.IDisposable" />
  public sealed class ZstandardDictionary : IDisposable
  {
    private byte[] dictionary;
    private IntPtr ddict;
    private Dictionary<int, IntPtr> cdicts = new Dictionary<int, IntPtr>();
    private object lockObject = new object();
    private bool isDisposed = false;

    /// <summary>
    /// Initializes a new instance of the <see cref="ZstandardDictionary"/> class.
    /// </summary>
    /// <param name="dictionary">The dictionary raw data.</param>
    public ZstandardDictionary(byte[] dictionary)
    {
      this.dictionary = dictionary;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ZstandardDictionary"/> class.
    /// </summary>
    /// <param name="dictionaryPath">The dictionary path.</param>
    public ZstandardDictionary(string dictionaryPath)
    {
      this.dictionary = File.ReadAllBytes(dictionaryPath);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ZstandardDictionary"/> class.
    /// </summary>
    /// <param name="dictionaryStream">The dictionary stream.</param>
    public ZstandardDictionary(Stream dictionaryStream)
    {
      using (var memoryStream = new MemoryStream())
      {
        dictionaryStream.CopyTo(memoryStream);
        this.dictionary = memoryStream.ToArray();
      }
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="ZstandardDictionary"/> class.
    /// </summary>
    ~ZstandardDictionary()
    {
      this.Dispose(false);
    }

    /// <summary>
    /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
    /// </summary>
    public void Dispose()
    {
      this.Dispose(true);
      GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases unmanaged resources.
    /// </summary>
    /// <param name="dispose"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
    private void Dispose(bool dispose)
    {
      if (this.isDisposed == false)
      {
        this.isDisposed = true;

        if (this.ddict != IntPtr.Zero)
        {
          Interop.ZSTD_freeDDict(this.ddict);
          this.ddict = IntPtr.Zero;
        }

        foreach (var kv in this.cdicts.ToList())
        {
          Interop.ZSTD_freeCDict(kv.Value);
          this.cdicts.Remove(kv.Key);
        }
      }
    }

    /// <summary>
    /// Gets the compression dictionary for the specified compression level.
    /// </summary>
    /// <param name="compressionLevel">The compression level.</param>
    /// <returns>
    /// The IntPtr to the compression dictionary.
    /// </returns>
    /// <exception cref="ObjectDisposedException">ZstandardDictionary</exception>
    internal IntPtr GetCompressionDictionary(int compressionLevel)
    {
      if (this.isDisposed)
      {
        throw new ObjectDisposedException(nameof(ZstandardDictionary));
      }

      lock (this.lockObject)
      {
        if (this.cdicts.TryGetValue(compressionLevel, out var cdict) == false)
        {
          this.cdicts[compressionLevel] = cdict = this.CreateCompressionDictionary(compressionLevel);
        }

        return cdict;
      }
    }

    /// <summary>
    /// Gets the decompression dictionary.
    /// </summary>
    /// <returns>
    /// The IntPtr to the decompression dictionary.
    /// </returns>
    /// <exception cref="ObjectDisposedException">ZstandardDictionary</exception>
    internal IntPtr GetDecompressionDictionary()
    {
      if (this.isDisposed)
      {
        throw new ObjectDisposedException(nameof(ZstandardDictionary));
      }

      lock (this.lockObject)
      {
        if (this.ddict == IntPtr.Zero)
        {
          this.ddict = this.CreateDecompressionDictionary();
        }

        return this.ddict;
      }
    }

    /// <summary>
    /// Creates a new compression dictionary.
    /// </summary>
    /// <param name="compressionLevel">The compression level.</param>
    /// <returns>
    /// The IntPtr to the compression dictionary.
    /// </returns>
    private IntPtr CreateCompressionDictionary(int compressionLevel)
    {
      var alloc = GCHandle.Alloc(this.dictionary, GCHandleType.Pinned);

      try
      {
        var dictBuffer = Marshal.UnsafeAddrOfPinnedArrayElement(this.dictionary, 0);
        var dictSize = new UIntPtr((uint)this.dictionary.Length);
        return Interop.ZSTD_createCDict(dictBuffer, dictSize, compressionLevel);
      }
      finally
      {
        alloc.Free();
      }
    }

    /// <summary>
    /// Creates a new decompression dictionary.
    /// </summary>
    /// <returns>
    /// The IntPtr to the decompression dictionary.
    /// </returns>
    private IntPtr CreateDecompressionDictionary()
    {
      var alloc = GCHandle.Alloc(this.dictionary, GCHandleType.Pinned);

      try
      {
        var dictBuffer = Marshal.UnsafeAddrOfPinnedArrayElement(this.dictionary, 0);
        var dictSize = new UIntPtr((uint)this.dictionary.Length);
        return Interop.ZSTD_createDDict(dictBuffer, dictSize);
      }
      finally
      {
        alloc.Free();
      }
    }
  }
}