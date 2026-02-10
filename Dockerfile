FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (ffmpeg + imagemagick for MoviePy rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    imagemagick \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick policy so MoviePy TextClip can render text
RUN if [ -f /etc/ImageMagick-6/policy.xml ]; then \
      sed -i 's/rights="none" pattern="@\*"/rights="read|write" pattern="@*"/' /etc/ImageMagick-6/policy.xml; \
    fi

# Copy dependency files and source for installation
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .
COPY alembic.ini ./
COPY migrations/ ./migrations/

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["uvicorn", "shorts_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
