FROM ros:noetic

RUN apt-get update && \
    apt-get install -y python-is-python3 python3-pip ros-noetic-cv-bridge && \
    pip3 install tensorflow && \
    pip3 install tensorflow-addons tensorflow-datasets tensorflow-hub && \
    pip3 install sklearn && \
    pip3 install openradar
