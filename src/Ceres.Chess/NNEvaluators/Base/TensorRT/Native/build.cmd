  cl /std:c++17 /O2 /MD /LD /EHsc ^
     /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" ^
     /I"D:\TensorRT10.14.1.48\include" ^
     TensorRTWrapper.cpp ^
     /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64" ^
           /LIBPATH:"D:\TensorRT10.14.1.48\lib" ^
           nvinfer_10.lib nvonnxparser_10.lib cudart.lib ^
     /OUT:TensorRTWrapper.dll
