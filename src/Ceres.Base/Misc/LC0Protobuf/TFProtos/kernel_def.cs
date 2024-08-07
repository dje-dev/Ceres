// This file was generated by a tool; you should avoid making direct changes.
// Consider using 'partial classes' to extend these types
// Input: kernel_def.proto

#pragma warning disable CS0612, CS1591, CS3021, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
namespace Tensorflow
{

    [global::ProtoBuf.ProtoContract()]
    public partial class KernelDef : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"op")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Op { get; set; } = "";

        [global::ProtoBuf.ProtoMember(2, Name = @"device_type")]
        [global::System.ComponentModel.DefaultValue("")]
        public string DeviceType { get; set; } = "";

        [global::ProtoBuf.ProtoMember(3, Name = @"constraint")]
        public global::System.Collections.Generic.List<AttrConstraint> Constraints { get; } = new global::System.Collections.Generic.List<AttrConstraint>();

        [global::ProtoBuf.ProtoMember(4, Name = @"host_memory_arg")]
        public global::System.Collections.Generic.List<string> HostMemoryArgs { get; } = new global::System.Collections.Generic.List<string>();

        [global::ProtoBuf.ProtoMember(5, Name = @"label")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Label { get; set; } = "";

        [global::ProtoBuf.ProtoContract()]
        public partial class AttrConstraint : global::ProtoBuf.IExtensible
        {
            private global::ProtoBuf.IExtension __pbn__extensionData;
            global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
                => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

            [global::ProtoBuf.ProtoMember(1, Name = @"name")]
            [global::System.ComponentModel.DefaultValue("")]
            public string Name { get; set; } = "";

            [global::ProtoBuf.ProtoMember(2, Name = @"allowed_values")]
            public AttrValue AllowedValues { get; set; }

        }

    }

}

#pragma warning restore CS0612, CS1591, CS3021, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
