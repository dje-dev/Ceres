// This file was generated by a tool; you should avoid making direct changes.
// Consider using 'partial classes' to extend these types
// Input: summary.proto

#pragma warning disable CS0612, CS1591, CS3021, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
namespace Tensorflow
{

    [global::ProtoBuf.ProtoContract()]
    public partial class SummaryDescription : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"type_hint")]
        [global::System.ComponentModel.DefaultValue("")]
        public string TypeHint { get; set; } = "";

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class HistogramProto : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"min")]
        public double Min { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"max")]
        public double Max { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"num")]
        public double Num { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"sum")]
        public double Sum { get; set; }

        [global::ProtoBuf.ProtoMember(5, Name = @"sum_squares")]
        public double SumSquares { get; set; }

        [global::ProtoBuf.ProtoMember(6, Name = @"bucket_limit", IsPacked = true)]
        public double[] BucketLimits { get; set; }

        [global::ProtoBuf.ProtoMember(7, Name = @"bucket", IsPacked = true)]
        public double[] Buckets { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class SummaryMetadata : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1)]
        public PluginData plugin_data { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"display_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string DisplayName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(3, Name = @"summary_description")]
        [global::System.ComponentModel.DefaultValue("")]
        public string SummaryDescription { get; set; } = "";

        [global::ProtoBuf.ProtoContract()]
        public partial class PluginData : global::ProtoBuf.IExtensible
        {
            private global::ProtoBuf.IExtension __pbn__extensionData;
            global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
                => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

            [global::ProtoBuf.ProtoMember(1, Name = @"plugin_name")]
            [global::System.ComponentModel.DefaultValue("")]
            public string PluginName { get; set; } = "";

            [global::ProtoBuf.ProtoMember(2, Name = @"content")]
            public byte[] Content { get; set; }

        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class Summary : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"value")]
        public global::System.Collections.Generic.List<Value> Values { get; } = new global::System.Collections.Generic.List<Value>();

        [global::ProtoBuf.ProtoContract()]
        public partial class Image : global::ProtoBuf.IExtensible
        {
            private global::ProtoBuf.IExtension __pbn__extensionData;
            global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
                => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

            [global::ProtoBuf.ProtoMember(1, Name = @"height")]
            public int Height { get; set; }

            [global::ProtoBuf.ProtoMember(2, Name = @"width")]
            public int Width { get; set; }

            [global::ProtoBuf.ProtoMember(3, Name = @"colorspace")]
            public int Colorspace { get; set; }

            [global::ProtoBuf.ProtoMember(4, Name = @"encoded_image_string")]
            public byte[] EncodedImageString { get; set; }

        }

        [global::ProtoBuf.ProtoContract()]
        public partial class Audio : global::ProtoBuf.IExtensible
        {
            private global::ProtoBuf.IExtension __pbn__extensionData;
            global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
                => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

            [global::ProtoBuf.ProtoMember(1, Name = @"sample_rate")]
            public float SampleRate { get; set; }

            [global::ProtoBuf.ProtoMember(2, Name = @"num_channels")]
            public long NumChannels { get; set; }

            [global::ProtoBuf.ProtoMember(3, Name = @"length_frames")]
            public long LengthFrames { get; set; }

            [global::ProtoBuf.ProtoMember(4, Name = @"encoded_audio_string")]
            public byte[] EncodedAudioString { get; set; }

            [global::ProtoBuf.ProtoMember(5, Name = @"content_type")]
            [global::System.ComponentModel.DefaultValue("")]
            public string ContentType { get; set; } = "";

        }

        [global::ProtoBuf.ProtoContract()]
        public partial class Value : global::ProtoBuf.IExtensible
        {
            private global::ProtoBuf.IExtension __pbn__extensionData;
            global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
                => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

            [global::ProtoBuf.ProtoMember(7, Name = @"node_name")]
            [global::System.ComponentModel.DefaultValue("")]
            public string NodeName { get; set; } = "";

            [global::ProtoBuf.ProtoMember(1, Name = @"tag")]
            [global::System.ComponentModel.DefaultValue("")]
            public string Tag { get; set; } = "";

            [global::ProtoBuf.ProtoMember(9, Name = @"metadata")]
            public SummaryMetadata Metadata { get; set; }

            [global::ProtoBuf.ProtoMember(2, Name = @"simple_value")]
            public float SimpleValue
            {
                get { return __pbn__value.Is(2) ? __pbn__value.Single : default; }
                set { __pbn__value = new global::ProtoBuf.DiscriminatedUnion32Object(2, value); }
            }
            public bool ShouldSerializeSimpleValue() => __pbn__value.Is(2);
            public void ResetSimpleValue() => global::ProtoBuf.DiscriminatedUnion32Object.Reset(ref __pbn__value, 2);

            private global::ProtoBuf.DiscriminatedUnion32Object __pbn__value;

            [global::ProtoBuf.ProtoMember(3, Name = @"obsolete_old_style_histogram")]
            public byte[] ObsoleteOldStyleHistogram
            {
                get { return __pbn__value.Is(3) ? ((byte[])__pbn__value.Object) : default; }
                set { __pbn__value = new global::ProtoBuf.DiscriminatedUnion32Object(3, value); }
            }
            public bool ShouldSerializeObsoleteOldStyleHistogram() => __pbn__value.Is(3);
            public void ResetObsoleteOldStyleHistogram() => global::ProtoBuf.DiscriminatedUnion32Object.Reset(ref __pbn__value, 3);

            [global::ProtoBuf.ProtoMember(4, Name = @"image")]
            public Summary.Image Image
            {
                get { return __pbn__value.Is(4) ? ((Summary.Image)__pbn__value.Object) : default; }
                set { __pbn__value = new global::ProtoBuf.DiscriminatedUnion32Object(4, value); }
            }
            public bool ShouldSerializeImage() => __pbn__value.Is(4);
            public void ResetImage() => global::ProtoBuf.DiscriminatedUnion32Object.Reset(ref __pbn__value, 4);

            [global::ProtoBuf.ProtoMember(5, Name = @"histo")]
            public HistogramProto Histo
            {
                get { return __pbn__value.Is(5) ? ((HistogramProto)__pbn__value.Object) : default; }
                set { __pbn__value = new global::ProtoBuf.DiscriminatedUnion32Object(5, value); }
            }
            public bool ShouldSerializeHisto() => __pbn__value.Is(5);
            public void ResetHisto() => global::ProtoBuf.DiscriminatedUnion32Object.Reset(ref __pbn__value, 5);

            [global::ProtoBuf.ProtoMember(6, Name = @"audio")]
            public Summary.Audio Audio
            {
                get { return __pbn__value.Is(6) ? ((Summary.Audio)__pbn__value.Object) : default; }
                set { __pbn__value = new global::ProtoBuf.DiscriminatedUnion32Object(6, value); }
            }
            public bool ShouldSerializeAudio() => __pbn__value.Is(6);
            public void ResetAudio() => global::ProtoBuf.DiscriminatedUnion32Object.Reset(ref __pbn__value, 6);

            [global::ProtoBuf.ProtoMember(8, Name = @"tensor")]
            public TensorProto Tensor
            {
                get { return __pbn__value.Is(8) ? ((TensorProto)__pbn__value.Object) : default; }
                set { __pbn__value = new global::ProtoBuf.DiscriminatedUnion32Object(8, value); }
            }
            public bool ShouldSerializeTensor() => __pbn__value.Is(8);
            public void ResetTensor() => global::ProtoBuf.DiscriminatedUnion32Object.Reset(ref __pbn__value, 8);

        }

    }

}

#pragma warning restore CS0612, CS1591, CS3021, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
