# Ceres - an MCTS chess engine for research and recreation

<img src="./images/CeresLogoLarge.png" width="436" height="275">

# Overview
Ceres is a strong UCI chess engine that uses the Monte Carlo Tree Search (MCTS) algorithm 
and deep neural networks. 

Importantly, Ceres also intended a comprehensive software platform for chess engine and neural network research,
designed for modularity, flexibility, and efficiency. 

## Status
The Ceres project has continued to progress since initial engine introduction at the beginning of 2021.

Ceres reached three milestones in late 2024:
* Publishing of multiple [neural networks](https://github.com/dje-dev/CeresNets) 
(of various sizes and speeds) trained using [CeresTrain](https://github.com/dje-dev/CeresTrain) on the [CeresNets](https://github.com/dje-dev/Ceres) repository.
* Publication of the first official engine release (1.0) on GitHub.
* First public participation in tournaments, including:
  - TCEC Swiss 7 (3rd place)
  - TCEC Cup 14 (4th place)
  - TCEC S27 Entrance League (1st place)
  - TCEC League 2 (1st place)

# Installation

[Installing](./markup/instructions_1.md) Ceres involves several steps:
* verifying prerequisites
* installing supporting libraries (Microsoft .NET, NVIDIA CuDNN)
* installing Ceres engine (either binary executable or building fron source)
* configuring Ceres engine
* (optional) configuring Ceres engine for highest performance (with the NVIDA TensorRT library)


## Ceres Neural Networks
The Ceres neural network architecture is similar to that of that [Lc0](<https://arxiv.org/abs/2409.12272>), featuring a postnorm encoder stack and multiple output heads (value, policy, ancillary). Importantly, Ceres copies extremely helpful the RPE (relative positional encoding) feature.

Extensive architecture search as been performed using PyTorch with the [CeresTrain training pipeline](<https://github.com/dje-dev/CeresTrain>) to try to identify improvements. The version 1.0 Ceres networks incorporate several additional features or adjustments that are believed to modestly boost performance and functionality:

1. The input preprocessing and output layers are simpler, with merely an embedding layer followed by normalization at input and simple FFN layers as output. The positional encoding takes the simple fo rm of one-hot vectors (for ranks and files).
2. An added  NonLinear Attention (NLA) feature augments the dot product self attention mechanism by adding an additional linear mappings connected by a nonlinearity in the preprocessing of the K, Q and V matrices. It closely follows the idea as proposed in ["NEURAL ATTENTION: ENHANCING QKV CALCULATION IN SELF-ATTENTION MECHANISM WITH NEURAL NETWORKS
"](https://arxiv.org/abs/2310.11398). [Tests](./images/NLA_performance.png) suggest it adds about 25 Elo for a cost of about 10% slowdown.
3. Initially the RPE feature of Lc0 was copied, except that only the K and Q parts are used (not V). 
Subsequently (with the 512x25 net) the RPE was replaced with the prior Smolgen technique of Lc0 due to higher inference speed, thanks to the analysis of Kovax.
4. A new second-order optimizer [SOAP](https://arxiv.org/abs/2409.11321) is used during training instead of Adam. Tests show that training convergence is sped up by about 30% (iterations) and 20% (wall-clock), with possibly superior performance at final convergence.
5. Experiments and also theoretical analysis from the literature suggest the current generation of chess transformers may not have 
the optimal shape, in particular they could benefit from greater depth relative to width. This seems intuitively plausible given the 
"lookahead nature" of a sequential decision making process like Chess. See: [Scale efficiently...](https://arxiv.org/abs/2109.10686)" and [Depth to Width Interplay...](https://arxiv.org/abs/2109.10686).
6. Auxiliary output heads of potential use for human interest or aiding search are available:
    * value head uncertainty
    * policy head uncertainty
    * projected future game score volatility




# Ceres Overview
Ceres ("Chess Engine for Research") is:
*  a state-of-the-art UCI-compliant chess engine employing the AlphaZero-style Monte Carlo Tree Search and deep neural networks
*  a flexible, modular and efficient software library with an exposed API to facilitate research in computer chess
*  a set of [integrated tools](./markup/Commands.md) for chess research (e.g for position analysis, suite testing, tournament manager)
*  ancillary features and applications to facilitate human analysis of chess positions, for example an integrated
[graphing](./markup/Graph.md) feature which generates on-the-fly visualizations of Ceres search results within a web browser,
or [game comparison](./markup/GameComp.md) feature which generates visualizations of differences between 2 or more games from a PGN file.

The Ceres MCTS engine is a novel implementation written in C# for the Microsoft .NET framework. This system 
comprises about 80,000 lines of source in 500 source code files, developed
as a way to try to make something good come of COVID confinement. The underlying
neural networks (and backend code to execute them) and backend code are currently mostly borrowed from the 
[LeelaChessZero project](https://lczero.org) via a "plug-in" architecture.

It is important to acknowledge that this project stands "on the shoulders of giants" - the pioneers
in the field such as DeepMind with their AlphaZero project and the ongoing Leela Chess Zero project. In 
some cases significant sections of code are largely based upon (transliterated from) other open source
projects such as Fathom (for tablebase access) or Leela Chess Zero (CUDA backend logic).

Although several fine open source implementations of MCTS engines are currently available, this project
is hoped to provide several important benefits:
* enhanced search speed, particularly on computers with multiple fast GPUs 
* a comprehensive API for chess research, rather than a narrow focus on a UCI engine
* an integrated set of tools (such as suite or tournament management) which simplify and accelerate testing of new research ideas
* an implementation using a modern programming language with features such as automatic garbage collections and development environments 
 that provide edit/compile/debug inner loops which are almost instantaneous
* a convenient testbed for implementing and evaluating potential new algorithmic innovations

## State of Development

Ceres was first released at the end of 2020 is still relatively early in its development.
Support for the neural network backend is current limited to CUDA-capable GPUs.

During 2020 numerous significant enhancements were made, including:
* added support for Linux operating system
* implemented the CUDA backend directly in C# (using transliteration and enhancement to the LC0 backend code, 
 including of the CUDA graphs feature for reduced inference latency)
* implemented C# tablebase probing logic (via a transliteration of the Fathom project code)
* added numerous algorithmic enhancements such as sibling blending and uncertainty boosting
* significantly improved CPU and memory efficiency, especially for very large searches

Ceres playing strength is currently competitive with top chess engines such as Leela Chess Zero and Stockfish,
depending of course considerably upon the particular types of hardware (CPU and GPU) available
for each engine.


## Ceres Software Architecture

The Ceres architecture is object oriented and organized into five layers:
* Ceres.Base - supporting algorithms and data structures not specific to the game of Chess
* Ceres.Chess - generic logic relating to the game of chess, such as move generation
* Ceres.MCTS - highly efficient implementation of Monte Carlo Tree Search
* Ceres.Features - implementations of various supplemental features such as suite and match play
* Ceres - top-level Console mode application with command line parsing to launch desired feature (or UCI engine)

The class library is intended to be reusable and offer comprehensive chess functionality 
to facilitate research, for example including:
* low-level chess concepts such as boards, moves, games, principal variations
* interfaces to external data such as PGN or EPD files, or local/remote neural network files
* a set of neural network position evaluators (including random, ensembled, roundrobin, split, or pooled)
* interfaces to external engines (via UCI) 
* integrated MCTS search engine allowing customization of parameters or introspection of computed search trees
* high-level modules for automated suite or tournament testing

The external API is not yet considered stable. Future effort will result in the publication
of documentation and more extensive code samples of the API, along with an overview
of the underlying data structures and algorithms.

As a teaser, the following [examples](./markup/APISamples.md)
demonstrate how the API can be leveraged to perform complex tasks using 
only a very small number of lines of code.


## Implementation Features

Numerous small to medium-sized features are believed to combine to help deliver 
strong performance of both search speed and play quality, including:

* A novel "dual CPUCT node selection algorithm" is used which alternates between two 
CPUCT levels (centered around the target CPUCT) on each batch, thereby minimizing collisions
(and allowing larger batches) in selected leafs and combining elements of 
greater breadth and depth simultaneously for improved play quality (suggested earlier
by LC0 contributor Naphthalin).

* MCTS leaf selection is highly parallelized in an almost lock-free way, 
with only a single descent and each visited node being visited at most once.

* MCTS leaf selection via PUCT algorithm is accelerated via SIMD hardware intrinsics (AVX),
which is made feasible by the above-mentioned parallelized descent algorithm.

* An overlapping execution approach allows efficient concurrent gathering and evaluation of batches.

* A relative virtual loss technique is used rather than absolute (to reduce the 
magnitude of distortions caused by node selection with virtual loss applied).

* The underlying search tree nodes are represented as fixed-size memory objects
which are cache-aligned and reserved up front by dynamically committed only as needed.
This structure enhances performance and facilitates efficient serialization of search tree state.
The data structures use 32-bit node indices instead of 64-bit pointers to reduce memory consumption
and make one-shot binary serialization possible. A set of "annotations"
are maintained for a cached subset of active nodes, containing derived
information useful in search.

* Transpositions are detected and short-circuited from needing neural network re-evaluation
by copying the neural networks from the nodes already "in situ" in the tree 
(thereby obviating explicit transposition tables or any limit on their size). A "virtual subtree"
techinque is used to avoid instantiating subtrees which are already transpositions
until they exceed 3 nodes in size, thereby improving efficiency and reducing memory requirements.

* Best move selection is often based on Q (subtree average evaluation) instead of N (number of visits).
Among other benefits, this opens the door to search methods more tuned to BAI (best arm identification) at the root.

* A "sibling blending" technique sometimes averages in information to newly visited nodes from their
siblings which have not yet been evaluated in the subtree but have already been evaluated
in other branches of the tree (i.e. are transpositions) thereby taking further advantage of the
substantial information captured in the full memory-resident search tree.

* An "uncertainty boosting" technique slightly incentivizes exploration at nodes 
with high uncertainty (historical variability of backed-up node evaluations), in the spirit
of the UCB algorithm's optimism (more variability might signal more potential upside,
and/or indicates that the marginal information gain of further visits is higher).

* Extensive use is made of fairly fine-grained parallelism to speed up many operations,
using the .NET Thread Parallel Library (TPL) for convenience and high efficiency.

* Critical components of the engine have been extensively optimized with
the help of tools such as Intel vTune and careful attention to processor 
details such as memory alignment, false sharing, NUMA effects, locking primitives, prefetching,
branch prediction, and maximizing instruction-level parallelism.

* The neural network evaluator framework is extensible with current implementations provided 
for random, CUDA using Leela Chess Zero networks, and an experimental NVIDIA Tensor RT backend
accepting ONNX network files, facilitating experimentation with alternate network architectures
or inference precisions (8 bit).

## Configuration and Installation

The setup [instructions](./markup/instructions_1.md) describe the sequence of steps
currently needed to install and configure Ceres. Although installation procedures
have been simplified since since first release, the process is not yet "single-click" easy
and does require several steps and careful attention.

As is typical of chess engines, no GUI is directly provided. Instead users typically
use GUI front-ends such as Arena, or the excellent Nibbler (https://github.com/rooklift/nibbler/releases) GUI which is optimized
for MCTS-style engine such as Ceres or Leela Chess Zero.


## Monitoring Tool
Event logging and statistics collection are very useful (and interesting) tools.
Ceres provides an optional realtime [monitoring](./markup/Monitoring.md) system.


## Contributing
It is hoped that Ceres will be a community effort. At this early stage, it probably 
does not make sense to be soliciting a large number of small improvements. Instead
it is suggested that contributions would be most useful in the following areas:

* testing of installation and operation on heterogeneous software/hardware environments
* testing against alpha/beta engines such as Stockfish
* feedback on the (limited) documentation provided so far
* opening issues to identify any bugs or anomalies noted in testing
* independent assesment of Ceres actual performance (speed and play quality)
* comments on the design and API surface
* suggestions for the most needed missing features

Somewhat bigger picture, thoughts and help with the architecture and implementation of backends would be 
especially welcomed. In particular, it is hoped to eventually generalize the interface between 
LC0 backends and arbitrary clients so this large and complex set of backends could be more 
widely leveraged by other chess engines including Ceres.


## Acknowledgements and Thanks

It goes without saying that Ceres builds on the work of many other researchers
and software developers, most notably:
* DeepMind's Alpha Zero project (for the basic architecture and search algorithms)
* Leela Chess Zero project for the neural networks and backends for inferencing with those networks
* Judd Niemann for the move generator code (translated and adapted from C++) (https://github.com/jniemann66/juddperft)
* Microsoft for the elegant C# language, performant .NET runtime, and excellent set of free software development tools such as Visual Studio

Special thanks are owed to to Kan for creating the Ceres logo as shown at the top of this page.

## License

Ceres is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 or later of the License, or
(at your option) any later version.

Ceres is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.


### Additional permission under GNU GPL version 3 section 7

_The source files of Ceres have the following additional permission, 
as allowed under GNU GPL version 3 section 7:_

If you modify this Program, or any covered work, by linking or
combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
modified version of those libraries), containing parts covered by the
terms of the respective license agreement, the licensors of this
Program grant you additional permission to convey the resulting work.

