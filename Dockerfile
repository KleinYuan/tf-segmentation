FROM tiangolo/uwsgi-nginx-flask:python2.7

RUN pip install -r requirements.txt

COPY . /app