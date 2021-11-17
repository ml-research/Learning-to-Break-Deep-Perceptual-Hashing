FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

ARG USER_ID
ARG GROUP_ID

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /code

COPY requirements.txt .

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN apt-get update && apt-get -y upgrade && apt-get -y install git nano
RUN pip install -r requirements.txt

USER user
