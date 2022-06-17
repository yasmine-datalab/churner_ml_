FROM --platform=linux/amd64 python:3.10-slim

COPY churner/app/requirements.txt /tmp

RUN apt-get update && apt-get -y install python3-pip && pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

#Create an user with minimal privileges
RUN groupadd -g 1000 mlgroup && useradd -u 1000 -g mlgroup mluser

RUN mkdir -p /usr/src/app/ && chown -R mluser:mlgroup /usr/src/app/

WORKDIR  /usr/src/app/

USER mluser:mlgroup

COPY --chown=mluser:mlgroup churner ./churner/

EXPOSE 5555

CMD ["python", "churner/app/api.py"]
