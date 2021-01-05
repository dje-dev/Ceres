#### Leela Chess Zero Plugin (LC0.DLL)

TLDR summary: Build LC0 from source, replacing just one file (network_cudnn.cc) 
with the version provided in Ceres source code tree, and change output target to DLL.

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
* the network_cudnn.cc file is modified to append code that
 exposes certain data structures and functions to Ceres 
for direct access via interop. This modified file can be found in the 
in the Ceres source code tree under \src\Ceres.Chess\NNEvaluators\LC0DLL\network_cudnn.cc.
* the project file configuration is changed to output a library (DLL) instead
 of an executable (EXE)

This LC0.DLL can then be placed in the LC0 directory with other binaries 
and will be loaded in-process by Ceres for network evaluation.

If LC0 building from source, more detailed instructions can be found in the Ceres source
code under the file named "building_LC0_EXE_and_DLL.txt."

Also, some Ceres users have offered the following additional tips:
* If you get build classes between backends, edit meson_options.txt to disable plain_cuda
* Verify you see that "Library cudnn found: YES" when you run build.cmd (otherwise it isn't going to build the new files)

 