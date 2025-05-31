# Usa un'immagine base con Python
FROM python:3.11-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia tutti i file nel container
COPY . .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta Streamlit
EXPOSE 8501

# Comando per avviare Streamlit
CMD ["streamlit", "run", "streamlit_finale.py", "--server.port=8501"]
