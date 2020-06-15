FROM ubuntu:focal
MAINTAINER Guillem Francès <guillem.frances@upf.edu>

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
	build-essential \
	cmake \
	ca-certificates \
	curl \
	python3 python3-pip python3-dev python3-setuptools \
	scons \
	git \
	pkg-config \
	libboost-program-options-dev libboost-filesystem-dev libboost-system-dev \
	libboost-chrono-dev libboost-timer-dev libboost-serialization-dev \
    && rm -rf /var/lib/apt/lists/*


# Set up some environment variables
ENV BASE_DIR "/code"
ENV CXX g++

# Install the wheel module for subsequents pip installs
RUN pip3 install wheel

RUN mkdir -p $BASE_DIR

#################################
# Install & build the FS planner
#################################
ENV FS_PATH "${BASE_DIR}/fs"
ADD fs $FS_PATH
RUN cd $FS_PATH && git submodule update --init && python3 ./build.py -p


#################################
# Install Tarski
#################################
ENV TARSKI_PATH "${BASE_DIR}/tarski"
ADD tarski $TARSKI_PATH
RUN cd $TARSKI_PATH && pip3 install -e .


#################################
# Install & build SLTP
#################################
ENV SLTP_PATH "${BASE_DIR}/sltp"
ADD sltp $SLTP_PATH
RUN cd $SLTP_PATH && pip3 install -e .
RUN cd $SLTP_PATH/src/features && cmake . && make -j4 && rm -rf CMakeFiles

#################################
# Install Basilisk
#################################
ENV BASILISK_PATH "${BASE_DIR}/basilisk"
ADD basilisk $BASILISK_PATH
RUN cd $BASILISK_PATH && pip3 install -e .


WORKDIR $SLTP_PATH
#ENTRYPOINT ["sh", "-c", "$SLTP_PATH/experiments/run.py"]
CMD ["bash"]

#COPY ./docker-entrypoint.sh /
#ENTRYPOINT ["/docker-entrypoint.sh"]
