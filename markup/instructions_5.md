# Run Ceres from Command Line - Basic Test

Because Ceres supports the UCI command line protocol, you can configure and run the engine from the command line to confirm correct installation.

1. Navigate to the directory containing the Ceres executable and launch Ceres. The Ceres banner should appear, indicating that the engine is ready to process UCI commands.

2. Configure the following UCI options:

   - **Device**: Defaults to `GPU:0`, indicating the GPU with index 0 should be used. To use multiple GPUs:
     ```
     setoption name device value GPU:0,1
     ```

   - **WeightsFile**: Specifies the neural network to use. There is no default value, so this must be set. For example, to use the smallest available Ceres network (dimension 256 with 10 layers):
     ```
     setoption name weightsfile value C1-256-10
     ```

3. Ceres will check if the ONNX file (e.g., `C1-256-10.onnx`) is in the current working directory. If not, it will attempt to download it from the [CeresNets GitHub site](https://github.com/dje-dev/CeresNets) into the current directory.

4. Once complete, Ceres is ready to perform searches. For example:
   ```
   go nodes 1000
   ```

[← Previous](instructions_4.md) | [Next →](instructions_6.md)
