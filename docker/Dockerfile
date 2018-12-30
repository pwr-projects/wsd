FROM clarinpl/nlptools:latest

RUN apt-get update
RUN apt-get install -y wget subversion software-properties-common python3 python3-pip

RUN pip3 install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN mkdir /project
WORKDIR /project

EXPOSE 8888

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
