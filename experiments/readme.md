
# Basilisk-NN

# Installation
Whole installation process is yet not fully automatized (sorry!), as some components
are C++ code that needs to be compiled outside pip. Easiest thing for now is to clone
manually the SLTP dependency  and to  `pip install -e .` it before running
`pip install -e .` in this repo.
Once that is done, some C++ in the SLTP project will need to be compiled, see below.
The exact commit ID to checkout from SLTP is on the `setup.py` file of this project. 


# Running the NN pipeline in Basilisk
New experiments are under `nn-experiments` folder.
All experiments go through a "dispatcher" `run.py` that reads particular experiment
configuration data from different files. So for instance running `./run.py gripper:sample01`
will go to python file `gripper.py` and look up there the experiment configuration labeled "sample01".

Chech the readme SLTP file at <https://github.com/aig-upf/sltp/tree/integrating-with-tarski>
for more details on how to install / run the whole thing.

So far, we can sample state space (with FS planner) and generate features (with SLTP C++ generator)
e.g. by running:  

    FS_PATH=/home/frances/projects/code/fs-sltp ./run.py gripper:sample01
    
This assume that you have first built the FS planner, which is under $FS_PATH, and 
also that you have built the C++ feature generator in SLTP, see instructions in readme above.