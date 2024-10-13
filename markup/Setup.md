## Setting up Ceres

Please note that currently the installation and configuration is somewhat involved.
This is expected to be simplified in the future.

#### Hardware/Software Requirements
* Windows 10 or Linux operating system
* Processor based on Intel/AMD x86 architecture with AVX support
* At least one NVIDIA GPU with CUDA device supporting compute capability level 7.0 or above (e.g. 2060 or above)
* Minimum 8GB of memory (recommended)

#### Installing Prerequisite of Microsoft .NET 5
* Under Windows or Linux install the .NET 5.0 runtime or SDK from https://dotnet.microsoft.com/download/dotnet/5.0
* Under Linux the following command may suffice: "sudo dnf install dotnet-sdk-5.0"


#### Building and Running from Binary Distribution Packages (on releases page)
* Download and unpack the ZIP file with the Ceres release
* Navigate to the directory "artifacts\release\net5.0" under the main Ceres directory
* Possibly configure the Ceres.json file as desired (to customize network, GPUs, tablebases, etc.). See below.
* Execute Ceres.exe.


#### Building and Running from Source Code (alternate approach)
Please follow these preliminary steps (as needed) for installing the Ceres development environment:
* Use git to download the Ceres code from github at https://github.com/dje-dev/ceres
* Download and install Microsoft Visual Studio (free Community Edition) from https://visualstudio.microsoft.com/downloads/
* You can now launch Visual Studio and open the solution file Ceres.sln under the src directory. 
For best performance, make sure to choose "Release" from the top toolbar (not "Debug").
You can then run Ceres by "Run without debugging" command (Ctrl-F5.).
* Alternately, from a command line in the src subdirectory run "dotnet build -c Release"
and then navigate to and run the Ceres.exe which will be built under artifcacts subdirectories.


#### Configuring Ceres
The final required step is to make sure a required configuration file with 
the name Ceres.json is present in the working directory from which Ceres is
launched (typically under a subdirectory with name of the form artifacts/release/net5.0).
This file contains various configuration entries (some required, some optional).
This file can be edited using a standard text editor to modify or add entries.

If Ceres launched run without a Ceres.json file in the working directory,
the user will be prompted to enter the values of 4 required configuration settings.
These will then be used to initialize a functional Ceres.json file, which 
can then be customzied at will using a text editor (or the SETOPT command in Ceres).
