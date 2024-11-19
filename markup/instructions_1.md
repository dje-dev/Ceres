# Confirm Prerequisites

Ceres requires the following minimum software/hardware configuration:

- **x86 processor** (AMD or Intel)
- **Windows 10/11** or **Linux** operating system (64-bit)
- **NVIDIA GPU** with compute capability level 7.0 or above (e.g., 20x0 or above)
- **CUDA version 12.x** such as 12.6

**Hint**: To verify that CUDA 12 is installed and visible on the system path, issue the following commands:

1. Open a new Windows console or terminal session.
2. Type the following commands:
   ```
   where cublas64_12.dll
   where cublasLT64_12.dll
   where cudart64_12.dll
   ```

If these files are not found and you are sure that CUDA 12.x is installed, the problem is likely that the path to CUDA DLLs is not on the Windows system path.

If you do not yet have CUDA 12.x installed, download and install 
from the [NVIDIA web site](https://developer.nvidia.com/cuda-downloads).


You can search your boot drive (typically **C:\**) to find the directory where the files are located.
Then, use the **"Edit the System Environment Variables"** applet in the Windows Control Panel to append an entry to the **"Path"** environment variable that references the directory.

← | [Next →](instructions_2.md)
