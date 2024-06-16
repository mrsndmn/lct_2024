FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

WORKDIR /lct_rutube

RUN apt update
RUN apt install -y ffmpeg git
RUN pip install git+https://github.com/mrsndmn/transformers
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./scripts ./scripts
COPY ./src ./src
COPY ./demo.py ./demo.py

ENV PYTHONPATH=.:./src

RUN mkdir -p /tmp/avmatcher/extracted_audios
RUN mkdir -p /tmp/avmatcher/normalized_videos

ENTRYPOINT streamlit run demo.py