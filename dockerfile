FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./model /code/model
COPY ./templates /code/templates
COPY ./app.py /code/app.py

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python", "app.py"]
