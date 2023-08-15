FROM python:3.11.4-slim

ENV WORKDIR /application/src/

WORKDIR ${WORKDIR}

RUN apt update &&\
    apt install -y gcc g++ &&\
    rm -rf /var/lib/apt/lists/*

ADD requirements_versions.txt .
RUN pip install -U pip &&\
    pip install -r requirements_versions.txt
# TODO: to move commands from launch.py =)

ADD . .
