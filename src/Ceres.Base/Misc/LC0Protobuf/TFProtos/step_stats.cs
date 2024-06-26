// This file was generated by a tool; you should avoid making direct changes.
// Consider using 'partial classes' to extend these types
// Input: step_stats.proto

#pragma warning disable CS0612, CS1591, CS3021, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
namespace Tensorflow
{

    [global::ProtoBuf.ProtoContract()]
    public partial class AllocationRecord : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"alloc_micros")]
        public long AllocMicros { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"alloc_bytes")]
        public long AllocBytes { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class AllocatorMemoryUsed : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"allocator_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string AllocatorName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(2, Name = @"total_bytes")]
        public long TotalBytes { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"peak_bytes")]
        public long PeakBytes { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"live_bytes")]
        public long LiveBytes { get; set; }

        [global::ProtoBuf.ProtoMember(6, Name = @"allocation_records")]
        public global::System.Collections.Generic.List<AllocationRecord> AllocationRecords { get; } = new global::System.Collections.Generic.List<AllocationRecord>();

        [global::ProtoBuf.ProtoMember(5, Name = @"allocator_bytes_in_use")]
        public long AllocatorBytesInUse { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class NodeOutput : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"slot")]
        public int Slot { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"tensor_description")]
        public TensorDescription TensorDescription { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class MemoryStats : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"temp_memory_size")]
        public long TempMemorySize { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"persistent_memory_size")]
        public long PersistentMemorySize { get; set; }

        [global::ProtoBuf.ProtoMember(5, Name = @"persistent_tensor_alloc_ids", IsPacked = true)]
        public long[] PersistentTensorAllocIds { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"device_temp_memory_size")]
        [global::System.Obsolete]
        public long DeviceTempMemorySize { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"device_persistent_memory_size")]
        [global::System.Obsolete]
        public long DevicePersistentMemorySize { get; set; }

        [global::ProtoBuf.ProtoMember(6, Name = @"device_persistent_tensor_alloc_ids")]
        [global::System.Obsolete]
        public long[] DevicePersistentTensorAllocIds { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class NodeExecStats : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"node_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string NodeName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(2, Name = @"all_start_micros")]
        public long AllStartMicros { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"op_start_rel_micros")]
        public long OpStartRelMicros { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"op_end_rel_micros")]
        public long OpEndRelMicros { get; set; }

        [global::ProtoBuf.ProtoMember(5, Name = @"all_end_rel_micros")]
        public long AllEndRelMicros { get; set; }

        [global::ProtoBuf.ProtoMember(6, Name = @"memory")]
        public global::System.Collections.Generic.List<AllocatorMemoryUsed> Memories { get; } = new global::System.Collections.Generic.List<AllocatorMemoryUsed>();

        [global::ProtoBuf.ProtoMember(7, Name = @"output")]
        public global::System.Collections.Generic.List<NodeOutput> Outputs { get; } = new global::System.Collections.Generic.List<NodeOutput>();

        [global::ProtoBuf.ProtoMember(8, Name = @"timeline_label")]
        [global::System.ComponentModel.DefaultValue("")]
        public string TimelineLabel { get; set; } = "";

        [global::ProtoBuf.ProtoMember(9, Name = @"scheduled_micros")]
        public long ScheduledMicros { get; set; }

        [global::ProtoBuf.ProtoMember(10, Name = @"thread_id")]
        public uint ThreadId { get; set; }

        [global::ProtoBuf.ProtoMember(11, Name = @"referenced_tensor")]
        public global::System.Collections.Generic.List<AllocationDescription> ReferencedTensors { get; } = new global::System.Collections.Generic.List<AllocationDescription>();

        [global::ProtoBuf.ProtoMember(12, Name = @"memory_stats")]
        public MemoryStats MemoryStats { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class DeviceStepStats : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"device")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Device { get; set; } = "";

        [global::ProtoBuf.ProtoMember(2, Name = @"node_stats")]
        public global::System.Collections.Generic.List<NodeExecStats> NodeStats { get; } = new global::System.Collections.Generic.List<NodeExecStats>();

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class StepStats : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"dev_stats")]
        public global::System.Collections.Generic.List<DeviceStepStats> DevStats { get; } = new global::System.Collections.Generic.List<DeviceStepStats>();

    }

}

#pragma warning restore CS0612, CS1591, CS3021, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
