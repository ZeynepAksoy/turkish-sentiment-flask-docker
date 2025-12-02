FROM python:3.10-slim

WORKDIR /app

# Gerekli sistem paketleri (pandas vs için güvenli)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flask uygulaması
COPY app ./app

# Eğitilmiş modeli ve veriyi imajın içine kopyala
COPY emotion_model.keras ./emotion_model.keras
COPY data ./data

EXPOSE 5001

CMD ["python", "-m", "app.app"]

