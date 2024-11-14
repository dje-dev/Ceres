# Download and Install/Build the Ceres Engine

There are two installation options available. The first (prepackaged binaries) is recommended for Windows users.

### Option 1: Precompiled Option (Windows Only)
- Download and unpack the installation package, then run `Ceres.exe`.

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
