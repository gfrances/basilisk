
# Concept Based Heuristics



## Development Notes


## Installation and Usage

To install in development mode, issue the following command on the root directory of the project:

    pip install -e . --process-dependency-links

This should install all the required dependencies.

## Usage

Individual experiments are on the `experiments` folder. See e.g. `experiments/gripper.py`
for an example. Currently we can invoke the configuration `prob01` of the experiment by issuing:

    experiments/gripper.py prob01 --all


### Software Requirements

* CPLEX (Python API)
  To install the Python API for CPLEX, go to the directory `/path/to/cplex/python` and run

	```bash
     python3 setup.py install
	 ```

* Python 3.6+ with the following dependencies
  - pip or pip3
  - setuptools
  - python-dev
  - numpy


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
