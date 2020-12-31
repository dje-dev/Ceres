#### Leela Chess Zero Plugin (LC0.DLL)

TLDR summary: Build LC0 from source, outputting a DLL with one file (network_cuda.cc) 
substituted from the Ceres source code tree.

A chess engine such as Ceres requires a network file with 
a set of weights and also "backend" code that can 
perform inference using this network specification.

Ceres is designed to support multiple pluggable networks and backends.
Currently the most well developed support is for the Leela Chess Zero 
networks and backend code. With great appreciation we leverage
this capability within Ceres.

This support takes the form of a custom library file (LC0.DLL) which 
can be built from the LC0 code base on github with just two 
minor modifications:
* the network_cuda.cc file is modified to append code that
 exposes certain data structures and functions to Ceres 
for direct access via interop. This modified file can be found in the 
in the Ceres source code tree or eventually  From the fork at 
https://github.com/dje-dev/lc0.
* the project file configuration is changed to output a library (DLL) instead
 of an executable (EXE)

This LC0.DLL can then be placed in the LC0 directory with other binaries 
and will be loaded in-process by Ceres for network evaluation.

If LC0 building from source, more detailed instructions can be found in the Ceres source
code under the file named "building_LC0_EXE_and_DLL.txt."


 