FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt requirements-cloud.txt ./
RUN pip install -r requirements-cloud.txt
COPY . .
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
# Render / Hugging Face Spaces / Cloud Run pass PORT; default 8501
ENV PORT=8501
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health')"
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
