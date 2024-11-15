# Install NVIDIA CuDNN Library

Ceres leverages the NVIDIA **CuDNN library** to increase the speed of neural network inference. **Version 9.3** is required:

- Download **cuDNN 9.x** (e.g., 9.5.1) from the [NVIDIA site](https://developer.nvidia.com/cudnn-downloads).
- Install cuDNN by executing the installer and following the on-screen prompts.
- Append the path to the cuDNN DLLs in the **System Environment Variables** "PATH" (e.g., `C:\Program Files\NVIDIA\CUDNN\v9.3\bin\12.6`).

**Hint**: To verify that CuDNN 9 is installed and visible on the system path, issue the following commands:

1. Open a new Windows console or terminal session.
2. Type the following commands:
   ```
   where cudnn_graph64_9.dll
   where cudnn64_9.dll
   ```

**Hint**: If you require assistance in editing the system path on Windows, see [this](https://www.c-sharpcorner.com/article/how-to-addedit-path-environment-variable-in-windows-11/)
useful web page of instructions.

[← Previous](instructions_2.md) | [Next →](instructions_4.md)
