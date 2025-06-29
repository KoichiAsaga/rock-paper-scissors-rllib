FROM python:3.10.15
RUN pip3 install --upgrade pip
COPY requirements.txt /work/
RUN pip install --prefer-binary --no-cache-dir -r /work/requirements.txt
RUN apt-get update && apt-get install -y curl jq 