FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD sh -c "python -c \"import os,urllib.request,sys; port=os.getenv('PORT','7860'); sys.exit(0) if urllib.request.urlopen(f'http://localhost:{port}/health').getcode()==200 else sys.exit(1)\""

CMD ["python", "-m", "server.app"]
