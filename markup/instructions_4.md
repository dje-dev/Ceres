# Download and Install/Build the Ceres Engine

There are two installation options available. 
Windows users are recommended to use the first (prepackaged binaries).

### Option 1: Precompiled Option (Windows Only)
- From the [Releases](https://github.com/dje-dev/Ceres/releases) page choose the desired (typically latest) release
- Download and unpack the ZIP file containing the Ceres.exe and supporting files
- Unpack this ZIP file (for example, Ceres_V1.0.zip) into a directory
- Locate and run the executable (Ceres.exe)

### Option 2: Build from Source
- Clone the repository from GitHub and build from source:
  ```
  git clone https://github.com/dje-dev/Ceres.git
  cd Ceres/src
  dotnet build -c Release
  cd ../artifacts/release/net8.0
  ceres
  ```

[← Previous](instructions_3.md) | [Next →](instructions_5.md)
