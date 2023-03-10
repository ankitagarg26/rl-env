# This code snippet was taken from https://github.com/TTitcombe/docker_openai_gym

# Use python3.7
# See https://docs.docker.com/samples/library/python/ for latest 
FROM python:3.7


# # Install base utilities
# RUN apt-get update

# # Install miniconda
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#      /bin/bash ~/miniconda.sh -b -p /opt/conda

# # Put conda in path so we can use conda activate
# ENV PATH=$CONDA_DIR/bin:$PATH

# Working directory is / by default. We explictly state it here for posterity
WORKDIR /

# Move the requirements file into the image
COPY requirements.txt /tmp/

# Upgrade pip3
RUN pip3 install --upgrade pip

# Install the python requirements on the image
RUN pip3 install --trusted-host pypi.python.org -r /tmp/requirements.txt

# Remove the requirements file - this is no longer needed
RUN rm /tmp/requirements.txt

# Create a directory in which we can do our work
RUN mkdir /home/minigrid/

# Set it as the working directory
WORKDIR /home/minigrid/

# Copy over the code
COPY ./ /home/minigrid/

# Copy over the start-up script
ADD startup_script.sh /usr/local/bin/startup_script.sh

# Give permissions to execute
RUN chmod 777 /usr/local/bin/startup_script.sh

# Set the display when we run the container. This allows us to record without the user needing to type anything explicitly
ENTRYPOINT ["/usr/local/bin/startup_script.sh"]