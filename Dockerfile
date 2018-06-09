FROM ubuntu
MAINTAINER Witold Bolt

RUN apt-get update && apt-get install -y git build-essential python3 python3-pip libgsl-dev
RUN pip3 install -U tensorflow numpy keras
RUN git clone https://github.com/houp/identify.git $HOME/identify 
WORKDIR $HOME/identify
RUN cd $HOME/identify && ./build_lib_gcc.sh python/libspatial

