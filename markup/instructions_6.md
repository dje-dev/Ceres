# Additional Ceres UCI Options and Features

### Example UCI Options

- **Set Path to Tablebase Files**: Ceres supports several UCI options, including the standardized command to set the path to tablebase files. For example:
  ```
  setoption name SyzygyPath value d:\Syzygy
  ```
  
- **Using Pre-Transformer Lc0 Networks**: Ceres also supports many of the pre-transformer Lc0 networks in their native `.pb` format, including those in the T70, T74, T75, T78, T80, and T81 series. To use these nets, just reference them as the WeightsFile:
  ```
  setoption name weightsfile value d:\nets\weights_run2_703810.pb.gz
  ```
  Support for more recent transformer Lc0 networks (e.g., BT4) is also possible, but requires a preprocessing step of converting the file to ONNX format.

### Configuration File

- A configuration file can be used to list options that are desired as persistent defaults. The file `Ceres.json` is auto-created upon the first execution of Ceres.
- Ceres reads this configuration file at startup, which contains entries for commonly used settings.
Initially, the file is initialized with the most common settings set to default values:
  ```json
  {
    "SyzygyPath": null,
    "DirCeresNetworks": ".",
    "DirLC0Networks": ".",
    "DefaultDeviceSpecString": "GPU:0"
  }
  ```
- If you wish to customize Ceres so she always starts with certain customized settings, 
set the desired values this Ceres.json file. For example:
  ```json
  {
    "SyzygyPath": "c:\\Syzygy",
    "DirCeresNetworks": ".",
    "DirLC0Networks": "c:\\lc0_nets",
    "DefaultDeviceSpecString": "gpu:0",
    "DefaultNetworkSpecString": "Ceres:C1-640-25",
  }
  ```

### Using Lc0 Networks
Ceres is capable of using Lc0 networks. 

Earlier Lc0 networks (pre-transformer) such as T78 or T81 nets can be directly used by specifying an network ID beginning with "LC0:" for example:
```
  "LC0:d:\nets\weights_run3_784984.gz"
```
Later Lc0 networks (full transformer) such as BT4 must first be converted to ONNX format using the conversion feature built into Lc0 for example:
```
  lc0 leela2onnx --input=d:\nets\BT4-1024x15x32h-swa-6147500.pb.gz --output=d:\nets\BT4-1024x15x32h-swa-6147500.pb.gz.onnx --onnx-data-type=f16
```
thereafter the generated ONNX file can be used with Ceres (using TensorRT) by using the prefix "ONNX_TRT", for example:
```
  ceres network=ONNX_TRT:d:\nets\BT4-1024x15x32h-swa-6147500.pb.gz.onnx device=GPU:0
```

The execution speed of Ceres using ONNX+TensorRT is typically similar to that of the Lc0 backend. For smaller nets it tends to be somewhat slower, 
whereas for large nets (such as BT4) Ceres is sometimes faster (7600 vs 7200 nps on an A6000 Ada GPU).

[← Previous](instructions_5.md) | [Next →](instructions_7.md)
