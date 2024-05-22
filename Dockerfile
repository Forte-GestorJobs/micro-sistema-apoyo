FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
#RUN pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

COPY /api/main.py /app

COPY micros-gestor-jobs.pem /app

EXPOSE 443

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "443", "--ssl-keyfile", "micros-gestor-jobs.pem", "--ssl-certfile", "micros-gestor-jobs.pem"]
