## Setting up Ceres

Please note that currently the installation and configuration is somewhat involved.
This is expected to be simplified in the future.

#### Hardware/Software Requirements
* Windows 10 (a Linux version is under development)
* Processor based on Intel/AMD x86 architecture
* Processor support for AVX instruction set
* At least one NVIDIA GPU with CUDA drivers installed
* Minimum 8GB of memory (recommended)

#### Preliminaries
Please follow these preliminary steps (as needed) for installing the Ceres development environment:
* Install Microsoft .NET 5.0 runtime or SDK from https://dotnet.microsoft.com/download/dotnet/5.0
* Download version v0.26.3 for CUDA backend of Leela Chess Zero (file lc0-v0.26.3-windows-gpu-nvidia-cuda.zip
 from https://github.com/LeelaChessZero/lc0/releases/tag/v0.26.3)
* Build or download a backend plug-in customized for Ceres (LC0.DLL) and 
 place this file in the same directory as the location where LC0 was installed above (see below for more details)
* Use git to download the Ceres code from github at https://github.com/dje-dev/ceres
* Download and install Microsoft Visual Studio (free Community Edition) from https://visualstudio.microsoft.com/downloads/

#### Buliding the LC0.DLL
If a pre-built LC0.DLL is not provided or available for the
version of CUDA installed on your computer it will be necessary to 
[build from source](BuildDLL.md).


#### Building Ceres
Now you can launch Visual Studio and open the solution file Ceres.sln under the src directory. 
For best performance, make sure to choose "Release" from the top toolbar (not "Debug").
You can then run Ceres by "Run without debugging" command (Ctrl-F5.).


#### Configuring Ceres
The final required step is to make sure a required configuration file with 
the name Ceres.json is present in the working directory from which Ceres is
launched (typically under a subdirectory with name of the form artifacts/release/net5.0).
This file contains various configuration entries (some required, some optional).

If Ceres launched run without a Ceres.json file in the working directory,
the user will be prompted to enter the values of 4 required configuration settings.
These will then be used to initialize a functional Ceres.json file, which 
can then be customized at will using a text editor (or the SETOPT command in Ceres).

The following is a complete [example](SetupExample.png) of this setup process, followed by a 
simple UCI search command using the newly configured settings.

