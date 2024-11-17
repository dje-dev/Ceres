# (Optional) Install NVIDIA TensorRT

Ceres can optionally leverage the NVIDIA TensorRT library to increase neural network evaluation speed.
The speed improvement in nodes per second is typically 1.5x to 2.0x for NVIDIA GPUs in the 30x0 series or later.
To install TensorRT:
  a. Download and install version 10x.from the NVIDIA web site at <https://developer.nvidia.com/tensorrt>. 
   a This may require creating an NVIDIA developer account, but this process is free and immediate. 
     Any version 10.x should work, (but only versions from 10.3 thru 10.6 have been tested).
  b. After download, unzip the compressed file to a location on your system.
  c. On Windows, the path to the DLLs files must be appended to the system path. For example if the files were downloaded to c:\TensorRT15 then the following PATH entry would be added:
       "C:\TensorRT15\lib"

By default Ceres uses the built-in CUDA execution engine. To instead use the faster TensorRT engine, modify the UCI device option by appending "#TensorRT16" as in:
```
setoption name device value GPU:0#TensorRT
```

Hint: If you wish to verify that TensorRT is installed and visible on the system path, issue the following commands and confirm the file is located. 
      Open a new Windows console or terminal session and type the following commands to confirm that 
      the necessary files are now visible on the system path:
```
  where nvinfer_10.dll
  where nvonnxparser_10.dll
```

Hint: When running on more recent NVIDIA GPUs, you should expect 1.5x to 2.0x raw neural network speed. This can be tested with the Ceres UCI command "backendbench".

Hint: NVIDIA TensorRT process builds engine files upon first reference to Ceres network. This process requires 30 to 90 seconds (first time only). Unfortunately, this process is not deterministic and 
occasionally (about 15% of the time) TensorRT will construct suboptimal engines which run only about 60% of the possible speed. If this happens, deleting the cached engine files and regenerating in Ceres usually fixes the problems.
For example, find the files under the trt_engines/<computer_name> subdirectory and delete them:
```
 Directory of C:\dev\Ceres\artifacts\release\net8.0\trt_engines\DEV
11/13/2024  09:58 PM        69,261,204 c1-256-10_229366337926929018_0_fp16_sm89.engine
11/13/2024  09:58 PM                28 c1-256-10_229366337926929018_0_fp16_sm89.profile
11/13/2024  09:58 PM         1,314,907 TensorrtExecutionProvider_cache_sm89.timing
```

Another way of determining if the engine build was optimal is to look at the file size. Suboptimal engine files are approximately 2x larger than the optimal. 
The [table](https://github.com/dje-dev/CeresNets) of Ceres nets lists the expected optimal sizes of engine files for each of the provided Ceres networks in the "TRT Size(mb)" column.

