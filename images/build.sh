#!/usr/bin/env bash
set -e -x

# This build script creates a tmp dir, copies all the required projects there, and
# creates a docker image there, plus a singularity imaged based on that.
# Everything could be perhaps more cleanly done without the script, just cloning
# all projects from a multistage docker build, but this script was written before
# multi-stage builds, and I've just adapted here for Basilisk.


BASEDIR=`pwd`/..

# the temp directory that we will use
WORK_DIR=`mktemp -d`

# check if tmp dir was created
if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

# deletes the temp directory
function cleanup {
  rm -rf "$WORK_DIR"
  echo "Deleted temp working directory $WORK_DIR"
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

pushd $WORK_DIR

# Get Tarski
git clone --depth 1 -b devel --single-branch git@github.com:aig-upf/tarski.git tarski

# Get the FS planner
git clone --depth 1 -b sltp-lite --single-branch git@github.com:aig-upf/fs-private.git fs
cd fs && git submodule update --init && cd ..

# Get current version of SLTP (from local directory)
git clone --depth 1 -b integrating-with-tarski --single-branch git@github.com:aig-upf/sltp.git sltp

# Copy
mkdir basilisk
cp -R $BASEDIR/src basilisk
cp -R $BASEDIR/setup.py basilisk
cp -R $BASEDIR/README.md basilisk
cp -R $BASEDIR/experiments basilisk

# Build Docker image
cp $BASEDIR/images/Dockerfile .
sudo docker build -t basilisk .
#cp $BASEDIR/images/Singularity .
#sudo singularity build basilisk.sif Singularity


# Upload image to the amazon cluster
# docker save sltp | bzip2 | pv | ssh awscluster 'bunzip2 | docker load'

# Upload image to Docker Hub
sudo docker tag basilisk:latest gfrancesm/basilisk:latest
echo "Docker image has been created. Run \"docker push gfrancesm/basilisk:latest\" if you want to push it to the registry"

# Create Singularity image from the Docker image just created
sudo singularity build basilisk.sif docker-daemon://basilisk:latest

mv basilisk.sif $BASEDIR/images

# Cleanup tmp directory and go back to original directory
cleanup
popd