FROM python:3.13-slim
WORKDIR /app
ARG NOERGLER_VERSION=dev
ENV NOERGLER_VERSION=${NOERGLER_VERSION}
COPY certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# o200k_base encoding — used by gpt-4o and gpt-5
RUN python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4o')"
COPY app/ app/
COPY prompts/ prompts/
COPY alembic.ini alembic.ini
COPY alembic/ alembic/
RUN chmod -R g=u /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
