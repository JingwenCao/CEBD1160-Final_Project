
FROM ubuntu:16.04
MAINTAINER Jingwen Cao <JingwenCao.github.io>

RUN apt-get update -qqq

RUN apt-get install -y python3-pip
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install sklearn

RUN mkdir -p /opt/

COPY breastcancer_analysis.py /opt/

ENTRYPOINT ["python3", "/opt/breastcancer_analysis.py"]

