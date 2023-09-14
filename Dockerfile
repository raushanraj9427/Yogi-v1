# Dockerfile Development

# pull the official docker image
FROM python:3.11.1-slim

# set work directory
WORKDIR /usr/src/app 

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# copy project
COPY . .

EXPOSE 8000