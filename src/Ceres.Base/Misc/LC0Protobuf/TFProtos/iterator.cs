// This file was generated by a tool; you should avoid making direct changes.
// Consider using 'partial classes' to extend these types
// Input: iterator.proto

#pragma warning disable CS0612, CS1591, CS3021, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
namespace Tensorflow
{

    [global::ProtoBuf.ProtoContract()]
    public partial class IteratorStateMetadata : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"version")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Version { get; set; } = "";

        [global::ProtoBuf.ProtoMember(2, Name = @"keys")]
        public global::System.Collections.Generic.List<string> Keys { get; } = new global::System.Collections.Generic.List<string>();

    }

}

#pragma warning restore CS0612, CS1591, CS3021, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
