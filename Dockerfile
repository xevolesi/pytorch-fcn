FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt update -y && apt upgrade -y

WORKDIR /fcn

# Install python3.10 and pip.
ENV PYTHON_GET_PIP_URL https://github.com/pypa/get-pip/raw/d5cb0afaf23b8520f1bbcfed521017b4a95f5c01/public/get-pip.py
RUN apt install software-properties-common wget -y \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install python3.10 python3.10-dev -y \
    && wget -O get-pip.py "$PYTHON_GET_PIP_URL" \
    && python3.10 get-pip.py \
		--disable-pip-version-check \
		--no-cache-dir \
		--no-compile \
    && python3.10 -m pip install --upgrade pip

# Install requirements.
COPY requirements.txt requirements.txt
RUN python3.10 -m pip install -r requirements.txt
COPY requirements.optional.txt requirements.optional.txt
RUN python3.10 -m pip install -r requirements.optional.txt

COPY . .
CMD [ "python3.10", "train.py" ]
