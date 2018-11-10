FROM ubuntu:xenial

RUN apt-get update
RUN apt-get install -y wget subversion software-properties-common python3=3.5* python3-pip
RUN pip3 install --upgrade pip

RUN wget -O - http://sgjp.pl/apt/sgjp.gpg.key| apt-key add -
RUN add-apt-repository 'deb http://sgjp.pl/apt/ubuntu-xenial xenial main'
RUN apt-get update

RUN apt-get install -y  libmorfeusz2 python3-morfeusz2

RUN mkdir -p /py_libs/plwn_api
RUN svn co -q http://svn.clarin-pl.eu/svn/ijn_students/plwn_api /py_libs/plwn_api

RUN pip3 install -e /py_libs/plwn_api

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN mkdir /project
WORKDIR /project

EXPOSE 8888

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]

