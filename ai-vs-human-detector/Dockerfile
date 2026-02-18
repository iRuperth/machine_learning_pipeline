# Usamos una imagen de Python ligera
FROM python:3.10-slim

# Instalamos uv directamente desde su instalador oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Evitamos archivos temporales de Python y activamos logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Copiamos los archivos de configuración de uv primero (para caché rápido)
COPY pyproject.toml uv.lock ./

# Instalamos las dependencias sin instalar el proyecto todavía
# Esto congela el entorno exacto que tenemos en la maquina
RUN uv sync --frozen --no-install-project

# Copiamos el resto de carpetas y archivos.
COPY . .

# Exponemos el puerto de FastAPI
EXPOSE 8000

# Arrancamos la API usando el entorno virtual gestionado por uv
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
