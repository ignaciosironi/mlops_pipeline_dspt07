# ---------------------------
# 1. Imagen base
# ---------------------------
FROM python:3.11-slim

# ---------------------------
# 2. Set working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# 3. Copiar requirements
# ---------------------------
COPY requirements.txt .

# ---------------------------
# 4. Instalar dependencias
# ---------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------
# 5. Copiar proyecto
# ---------------------------
COPY . .

# ---------------------------
# 6. Exponer puerto
# ---------------------------
EXPOSE 8000

# ---------------------------
# 7. Comando de arranque
# ---------------------------
CMD ["uvicorn", "src.model_deploy_avance4:app", "--host", "0.0.0.0", "--port", "8000"]