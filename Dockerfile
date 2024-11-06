FROM ubuntu:latest
LABEL authors="bobin"
FROM python:3.11
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["fastapi", "run", "api/api.py"]
ENTRYPOINT ["top", "-b"]