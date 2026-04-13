FROM python:3.12-slim
WORKDIR /app
COPY certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4')"
COPY app/ app/
COPY prompts/ prompts/
COPY alembic.ini alembic.ini
COPY alembic/ alembic/
RUN chmod -R g=u /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
