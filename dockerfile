FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./model /code/model
COPY ./templates /code/templates
COPY ./app.py /code/app.py

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app", "--reload", "--workers=1"] && python -mwebbrowser http://0.0.0.0:5000
