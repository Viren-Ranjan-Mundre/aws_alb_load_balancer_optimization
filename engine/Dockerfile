FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY optimizer.py .
COPY rl_agent.py .
COPY risk_model.pkl .
CMD ["uvicorn", "optimizer:app", "--host", "0.0.0.0", "--port", "8000"]
