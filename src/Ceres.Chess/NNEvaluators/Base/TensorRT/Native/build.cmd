  @echo off
  setlocal enabledelayedexpansion

  REM ============================================================================
  REM Configuration - Update these paths for your environment
  REM ============================================================================
  set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
  set "TENSORRT_ROOT=D:\TensorRT10.14.1.48"

  REM Derived paths
  set "CUDA_INCLUDE=%CUDA_ROOT%\include"
  set "CUDA_LIB=%CUDA_ROOT%\lib\x64"
  set "TENSORRT_INCLUDE=%TENSORRT_ROOT%\include"
  set "TENSORRT_LIB=%TENSORRT_ROOT%\lib"

  REM Output configuration
  set "OUTPUT_DLL=TensorRTWrapper.dll"
  set "SOURCE_FILE=TensorRTWrapper.cpp"

  REM ============================================================================
  REM Validate directories exist
  REM ============================================================================
  echo Validating build environment...

  if not exist "%CUDA_ROOT%" (
      echo ERROR: CUDA directory not found: %CUDA_ROOT%
      echo Please update CUDA_ROOT in this script to match your installation.
      exit /b 1
  )

  if not exist "%CUDA_INCLUDE%" (
      echo ERROR: CUDA include directory not found: %CUDA_INCLUDE%
      exit /b 1
  )

  if not exist "%CUDA_LIB%" (
      echo ERROR: CUDA lib directory not found: %CUDA_LIB%
      exit /b 1
  )

  if not exist "%TENSORRT_ROOT%" (
      echo ERROR: TensorRT directory not found: %TENSORRT_ROOT%
      echo Please update TENSORRT_ROOT in this script to match your installation.
      exit /b 1
  )

  if not exist "%TENSORRT_INCLUDE%" (
      echo ERROR: TensorRT include directory not found: %TENSORRT_INCLUDE%
      exit /b 1
  )

  if not exist "%TENSORRT_LIB%" (
      echo ERROR: TensorRT lib directory not found: %TENSORRT_LIB%
      exit /b 1
  )

  if not exist "%SOURCE_FILE%" (
      echo ERROR: Source file not found: %SOURCE_FILE%
      echo Please run this script from the directory containing %SOURCE_FILE%
      exit /b 1
  )

  REM ============================================================================
  REM Check for Visual Studio compiler
  REM ============================================================================
  where cl >nul 2>&1
  if errorlevel 1 (
      echo ERROR: cl.exe not found in PATH.
      echo Please run this script from a Visual Studio Developer Command Prompt.
      exit /b 1
  )

  echo All dependencies found. Starting build...
  echo.

  REM ============================================================================
  REM Build the DLL
  REM ============================================================================
  cl /std:c++17 /O2 /MD /LD /EHsc ^
     /I"%CUDA_INCLUDE%" ^
     /I"%TENSORRT_INCLUDE%" ^
     "%SOURCE_FILE%" ^
     /link /LIBPATH:"%CUDA_LIB%" ^
           /LIBPATH:"%TENSORRT_LIB%" ^
           nvinfer_10.lib nvonnxparser_10.lib cudart.lib ^
     /OUT:"%OUTPUT_DLL%"

  if errorlevel 1 (
      echo.
      echo ERROR: Build failed!
      exit /b 1
  )

  echo.
  echo Build successful: %OUTPUT_DLL%
  echo.

  endlocal
