FROM python:3.10.5

WORKDIR .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
ADD requirements.in .
RUN pip install -r requirements.in
ADD infer.py .
ADD ./tables/ ./tables
ADD ./uploads/ ./uploads

CMD [ "python","infer.py" ]
