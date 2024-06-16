FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

WORKDIR /lct_rutube

RUN apt update
RUN apt install -y ffmpeg git
RUN pip install git+https://github.com/mrsndmn/transformers
COPY requirements.txt ./
RUN pip install -r requirements.txt

