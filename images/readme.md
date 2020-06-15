
# Building a Singularity for Basilisk
The 



# Running Basilisk from the image
To run the image, we need to (1) mount the directory where the benchmarks are in the cluster
into the Singularity container, and (2) mount a directory with write permission into the container
as well, which we'll be used as Basilisk's workspace. This sample call will run the "small" experiment
from Gripper:

    singularity run  --bind /home/frances/projects/code/basilisk/benchmarks:/code/basilisk/benchmarks \
                     --bind .:/workspace ./basilisk.sif \
                     /code/basilisk/experiments/run.py --workspace /workspace gripper:small 

With the call above, all the output of the pipeline will be left in the current directory, since we're mounting
that directory as the container's `/workspace` directory. Of course you can change the `--bind` instruction
to put that somewhere else.