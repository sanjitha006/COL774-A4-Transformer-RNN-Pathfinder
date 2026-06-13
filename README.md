# Sequence-to-Sequence Maze Pathfinder

**Language & Method:** This project is implemented in Python utilizing PyTorch to build sequence-to-sequence neural networks. The implementation strictly permits libraries like pandas, numpy, matplotlib, argparse, and sys, but explicitly forbids scikit-learn or any AI-generation tools. The core architectures include an RNN Encoder-Decoder with Bahdanau Attention and a full Transformer Encoder-Decoder model.

**Motive & Objective:** The primary goal is to solve spatial reasoning tasks by training models to predict an optimal path through 6x6 mazes. The models process tokenized input sequences representing the maze's adjacency list, starting origin, and target destination, to sequentially generate the correct grid coordinates connecting them. The networks must successfully navigate both unambiguous fork-less mazes and complex forked mazes that require decision-making.

**Execution Constraints:** The training process must properly manage variable-length input sequences using techniques like padding or masking. For inference, the project utilizes an `eval.py` script requiring two command-line arguments: the path to a pre-trained model and the path to a `.txt` file containing the maze input. 
