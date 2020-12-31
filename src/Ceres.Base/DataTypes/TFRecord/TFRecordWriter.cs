#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using Ceres.Base.DataType;
using System.IO;
using Tensorflow;
using ProtoBuf;
using System;
using Ceres.Base.Benchmarking;

#endregion

// See: https://github.com/kevinskii/TFRecord.NET
// MIT License.

namespace Ceres.Base.DataTypes.TFRecord
{
  // Adapted from the Java implementation in:
  // https://github.com/tensorflow/ecosystem/blob/master/hadoop/src/main/java/org/tensorflow/hadoop/util/TFRecordWriter.java
  public class TFRecordWriter
  {
    private const uint MASK_DELTA = 0xa282ead8;
    private readonly Stream _stream;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="s">Stream to write TFRecord entries to</param>
    public TFRecordWriter(Stream s)
    {
      _stream = s;
    }

    public void Write(byte[] data)
    {
      // tfrecord format:
      //
      // uint64 length
      // uint32 masked_crc32_of_length
      // byte data[length]
      // uint32 masked_crc32_of_data
      byte[] len = ToUInt64LittleEndian((ulong)data.Length);
      _stream.Write(len, 0, len.Length);
      byte[] lenCRC = ToUInt32LittleEndian(MaskedCRC32c(len));
      _stream.Write(lenCRC, 0, lenCRC.Length);
      _stream.Write(data, 0, data.Length);
      byte[] dataCRC = ToUInt32LittleEndian(MaskedCRC32c(data));
      _stream.Write(dataCRC, 0, dataCRC.Length);
    }

    private uint MaskedCRC32c(byte[] data)
    {
      uint crc = Crc32CAlgorithm.Compute(data);
      return ((crc >> 15) | (crc << 17)) + MASK_DELTA;
    }

    // Converts an integer to a little endian byte array. Based partly on this Stack Overflow answer:
    // https://stackoverflow.com/a/2350112/353308
    private byte[] ToUInt64LittleEndian(ulong length)
    {
      byte[] bytes = new byte[sizeof(ulong)];
      for (int i = 0; i < bytes.Length; ++i)
      {
        bytes[i] = (byte)((length >> (8 * i)) & 0xFF);
      }
      return bytes;
    }

    private byte[] ToUInt32LittleEndian(uint length)
    {
      byte[] bytes = new byte[sizeof(uint)];
      for (int i = 0; i < bytes.Length; ++i)
      {
        bytes[i] = (byte)(((uint)length >> (8 * i)) & 0xFF);
      }
      return bytes;
    }

    #region Test

    public static void Test()
    {
      Feature featureBoardBitmaps = new Feature { Int64List = new Int64List() };
      Feature featureBoardValues = new Feature { FloatList = new FloatList() };
      Feature featurePolicies = new Feature { FloatList = new FloatList() };
      Feature featureValues = new Feature { FloatList = new FloatList() };

      Example example = new Example() { Features = new Features() };
      example.Features.feature.Add("board_bitmaps", featureBoardBitmaps);
      example.Features.feature.Add("board_values", featureBoardValues);
      example.Features.feature.Add("policies", featurePolicies);
      example.Features.feature.Add("values", featureValues);

      const int BATCH_SIZE = 1024;
      const int SIZE = BATCH_SIZE * 21 * 21 * 19;

      featureBoardBitmaps.Int64List.Values = new long[19*19*21];
      featureBoardValues.FloatList.Values = new float[19 * 19 * 21];
      featurePolicies.FloatList.Values = new float[19 * 21];
      featureValues.FloatList.Values = new float[1];

      featureBoardBitmaps.Int64List.Values[0] = 4;
//      featureBoardValues.FloatList.Values[1] = 3;
      featureValues.FloatList.Values[0] = 9;
//      featureValues.FloatList.Values[1] = 9.7f;

      using (new TimingBlock("Write TFRecord file"))
      {
        using (var outFile = File.Create(@"c:\temp\tfrecord.dat"))
        {
          var writer = new TFRecordWriter(outFile);

          for (int i = 0; i < BATCH_SIZE; i++)
          {
            var bytes = SerializationUtils.ProtoSerialize<Example>(example);
            writer.Write(bytes);
          }
          //File.WriteAllBytes("tfrecord.dat", xx);
        }
      }

      #endregion
    }

  }
}

/*
#if NOTX

# code to read a TFRecord file (tfrecord.dat)
# in Python to verify valid

import tensorflow as tf

feature_description = {
  "board_bitmaps":tf.io.FixedLenFeature([], tf.int64, default_value=0),
  "board_values":tf.io.FixedLenFeature([], tf.float32, default_value =0),
  "policies":tf.io.FixedLenFeature([], tf.float32, default_value=0),
  "values":tf.io.FixedLenFeature([], tf.float32, default_value =0),
}

raw_dataset = tf.data.TFRecordDataset('c:/temp/tfrecord.dat')

for item in raw_dataset:
# example = tf.io.parse_single_example(item, feature_description)
# print(example)
   print(item)

-----------------------------------------------------
# sample to create a tfrecord.dat file using Python
# which is known to be valid (for comparision purposes)
import tensorflow as tf

movie_name_list = tf.train.Int64List(value=[4,3])
movie_rating_list = tf.train.FloatList(value=[9.0, 9.7])

movie_names = tf.train.Feature(int64_list=movie_name_list)
movie_ratings = tf.train.Feature(float_list=movie_rating_list)

movie_dict = {
  'boards': movie_names,
  'values': movie_ratings
}

movies = tf.train.Features(feature=movie_dict)
example = tf.train.Example(features=movies)

with tf.io.TFRecordWriter('c:/temp/tfrecord.dat') as writer:
  writer.write(example.SerializeToString())


#endif
*/