# 1. Utiliser une image de base officielle de Python
FROM python:3.11-slim

# 2. Définir le répertoire de travail dans le conteneur
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copier le fichier des dépendances et installer les packages nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier les fichiers de l'application dans le conteneur
COPY api/ ./api/
COPY data/ ./data/

# 7. Définir le point d'entrée pour le conteneur
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]