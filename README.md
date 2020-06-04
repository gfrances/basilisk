
# Basilisk-NN

## Installation
Whole installation process is yet not fully automatized (sorry!), as some components
are C++ code that needs to be compiled outside pip. Easiest thing for now is to clone
manually the SLTP dependency  and to  `pip install -e .` it before running
`pip install -e .` in this repo.
Once that is done, some C++ in the SLTP project will need to be compiled, see below.
The exact commit ID to checkout from SLTP is on the `setup.py` file of this project. 


## Running the NN pipeline in Basilisk
All experiments are in the `experiments` folder.
All experiments go through a "dispatcher" `run.py` script that reads the particular experiment
configuration data from a single file. So for instance running `./run.py gripper:sample01`
will go to python file `gripper.py` and look up there the experiment configuration labeled "sample01".

Experiments are currently divided in two main blocks: 
1. Generation of training data.
1. Learning of a generalized heuristic.

This is so simply because Augusto is too lazy to install the necessary requirements for the training
data generation in the cluster, so this way we can generate training data locally, sync with the cluster,
and trigger the computationally-demanding learning part in the cluster.

The above split is reflected in the existence of two main scripts in the `experiments` directory:
1. `run.py`: In charge of generating the training data and serializing it to disk.
1. `learn.py`: Learning the generalized heuristic.


Chech the readme SLTP file at <https://github.com/aig-upf/sltp/tree/integrating-with-tarski>
for more details on how to install / run the whole thing.

So far, we can sample state space (with FS planner) and generate features (with SLTP C++ generator)
e.g. by running:  

    FS_PATH=/home/frances/projects/code/fs-sltp ./run.py gripper:sample01
    
This assume that you have first built the FS planner, which is under $FS_PATH, and 
also that you have built the C++ feature generator in SLTP, see instructions in readme above.

## Example Run
A simple run of the pipeline on the Gripper domain:
    
    # Generate training data:
    .../basilisk/experiments$ ./run.py gripper-m:learn
    
    # Train the NN to learn the heuristic:
    .../basilisk/experiments$ ./learn.py gripper-m:learn


## Software Requirements

* Python 3.6+ with the following dependencies
  - `keras`
  - `numpy`
  - `pip` or `pip3`
  - `python-dev`
  - `setuptools`
  - `sklearn`
  - `tensorflow`


## Domains and Instances
* Typed domains In order to use new domains with types, the order of the types
  in the domain file should be in decreasing order. In other words, it means
  that the "supertypes" should be declared first. See the Spanner domains as an
  example.

* Right now, we have the following domains:
  - `blocks`: working
  - `gripper`: (IPC version) working
  - `gripper-m`: (modified version) working
  - `spanner-ipc2011`: not working
  - `spanner`: (our instances) working
  - `visitall`: working
