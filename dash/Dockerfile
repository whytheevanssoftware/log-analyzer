# set base image (host OS)
FROM python:3.9

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

WORKDIR /code

RUN mkdir -p code/results

COPY code/ ./dash

# command to run on container start
CMD [ "python", "dash/app.py" ]