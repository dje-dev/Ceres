# Ceres Graphing Feature

## Introduction

**NOTE: This feature is current in beta test and requires building from source code. Only Windows has been tested so far. Feedback is welcomed.**

The Ceres graph visualization feature facilitates analysis of chess positions by graphically showing the main lines (most likely play continuations) arising from any Ceres search, including the actual board images and associated annotation. This capability is made possible by building on top of "Graphviz", a  widely-used cross-platform open source software package.

MCTS engines such as Ceres are especially well suited to generating such analysis (in a way that traditional alphabeta engines are not). In particular, as a natural part of their search algorithm they explore and create a durable memory representation of each subtree and its aggregate characateristics (such as evaluation). Furthermore, they produce metrics such as "percentage of visits from parent" and "percentage of visits in tree" which provide intuitive guidance about the magnitude of relevance for each position.


## Prerequsites
There are two preparatory steps required to use this feature:
- install the Graphviz software from https://graphviz.org/download/ (typically using the linked "graphviz-2.50.0 (64-bit) EXE installer"). When prompted, it is suggested to set the PATH for all users.
- add a new line to Ceres.json referencing the location of these binaries. For example:
```
"DirGraphvizBinaries":"C:\Program Files (x86)\Graphviz\bin"
```

## Generating Graphs
The graph feature is triggered by issuing the command "graph" at the UCI prompt. Optionally this can be followed by a detail level running from 0 (least detailed) thru 9 (most detailed).  Ceres will then generate the graph and launch browser to view. 
Graph generation typically requires between 1 and 3 seconds. For example:
```
go nodes 100_000

info depth 1 seldepth 1 time 87 nodes 1 score cp 4 tbhits 0 nps 11 pv  string M= NaN
info depth 11 seldepth 28 time 775 nodes 99988 score cp 13 tbhits 0 nps 128961 pv e2e4 c7c5 g1f3 e7e6 b1c3 b8c6 d2d4 c5d4 f3d4 string M= NaN
bestmove e2e4

graph 7
```

## Graph Output Examples

**Graph at lowest level of detail (0):**
![](./images/graph_0_example.PNG)

**Graph at highest level of detail (partial) (9):**
![](./images/graph_9_example.PNG)

## Tips on Interpreting the Graph
- the root position appears on the left followed by one or more "top moves" from this position (each as a boxed subgraph) with the "best move" shown in a slight green background color

- the lines with arrows show the moves that transition from one board to another, with the principal variation shown using a red lines

- the piece being moved is shown using yellow for the origin square and green for the destination square

- hovering over a board with the mouse will show a tooltip with additional details about the main possible continuation moves from that position, for example:
![](./images/graph_tooltip.PNG)

- clicking on a board will launch the position in lichess.com to facilitate sharing, further analysis, or extracting the corresponding FEN

- most browsers allow easy zoom in or out by holding down the Control key and using the mouse wheel


## Tips on Using the Graph Feature from other GUIs
This feature can be used in conjunction with any other GUI (such as Nibbler, Arena, Fritz, Chessbase, or BanksiaGUI). At any time while running an analysis using Ceres within the GUI, simply run Ceres with the graph command as the command line argument verb, such as:
```
ceres graph options=7
```

This will use interprocess communication to send a signal to the other Ceres instance running inside the GUI to pause and generate the the graph and show in the browser. NOTE: for this to work, there must be only one Ceres.exe currently running on the computer at this time.

Most GUIs also provide shortcuts for issuing UCI commands to engines without leaving the GUI application. Examples for Nibbler and Arena are shown here.

![](./images/graph_nibbler.PNG)
![](./images/graph_arena.PNG)

## Adding Reference Engine Analysis
Often it can be helpful to evaluate chess positions using more than one engine to get alternative
opinions of the position and associated best move. For example:
 -  analysis comparing two neural networks (both using Ceres) two understand different styles and strengths of various networks, or
 -  analysis comparing Ceres and LC0 to understand the differences between the two MCTS engines, or
 -  analysis comparing Ceres with an alphabeta engine such as Stockfish using a fundamentally different algorithm (minimax)

The graphing feature supports specifying an additional reference engine in the Ceres.json 
which will be called to evaluate every position in the diagram and then show the output of this analysis (score and best move) in an additional
line below the associated position. For example, in the example below Stockfish is configured as the reference engine
and a supplemental line appears at the bottom of the position with the corresponding analysis:
![](./images/graph_reference_engine.PNG)

In this case we see the Stockfish evaluation is much better than Ceres (280cp vs 81cp). We see Stockfish
suggested best move (Bg5) differs from the Ceres recommended best move (b4), and for this reason 
the line is shown with a red background. The "(+72 cp)" at the end appears because the best moves differed, 
and indicates that the Stockfish evaluates its best move to be superior to Ceres best move by 72 centipawns.

This graphing feature is activated at the UCI command prompt by specifying the 
string "ref" followed by a numerical multiplier after the graph command. For example:
```
graph 5 ref 2
```
requests a graph at detail level 5 with the reference engine also included and assigned a search time 2x that
of what was spent by the primary engine (Ceres) on each of the positions appearing in the graph.

Three different types of reference engines are supported. The first type is general UCI engine (such as Stockfish)
which is specified in Ceres.json along with optional number of threads and hash size in megabytes. The tablebase
are also set if standard Syzygy setting in Ceres is specified. For example:
```
"RefEngineEXE": "c:/engines/stockfish_14.1_win_x64_avx2.exe",
"RefEngineThreads": 32,
"RefEngineHashMB": 8192,
"SyzygyPath": "c:/sygyzy/5and6man",
```

The second type specifies the LC0 engine. The DirLC0Binaries directory will be
used to find the LC0.EXE, and LC0 will be configured with standard settings to match the Ceres configuration
(for example, the same neural network and GPUs).
```
"RefEngineEXE": "lc0",
"DirLC0Binaries":"c/engines/lc0",
```

The third type specifies the "Ceres" engine, which will simply run another search using Ceres but optionally
using a specified alternate neural network. For example:
```
"RefEngineEXE": "ceres",
"RefEngineNetworkFile":"c:/weights/lczero.org/weights_run2_790564.pb",
```

The reference engine multiplier behaves different depending on the type:
- for general UCI engine, the search time on each position is calibrated to be 
approximately a multiple of the search effort spent on by the primary Ceres engine. For example, if 20% 
of the nodes were spent on the position by Ceres, then 20% of the search time would be spent with the reference engine.
- for MCTS engines (LC0 or Ceres) the number of target search visits will be set to exactly match 
some multiple of the number of visits achieved by the primary Ceres engine

## Method of Operation

The high-level steps taken by Ceres to generate and show the graph are:
* traverse the search tree and pick which nodes to include (only this which have sufficient fraction of visits relative to both the tree as a whole and its parent)

* emit a text file within a temporary directory using the ".dot" format (domain specific markup language used by Graphviz for defining graphs) which specifies the graph structure and associated attributes (such as titles and tooltips and formatting)

* (each board image is itself a separate .svg file generated in the same temporary directory and referenced by the primary .svg)

* execute the DOT executable (part of Graphviz package) as a background process to convert the .dot file into a corresponding .svg file (a widely used graphics markup language that browsers can render natively)

* launch the browser specifying the generated .svg file as an argument
