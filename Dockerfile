FROM python:3.10-slim

# Install LaTeX (for matplotlib usetex) and OpenCV runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-fonts-recommended \
        cm-super \
        dvipng \
        git \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (copy only what pip needs so this layer is cached
# unless dependencies change)
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
RUN pip install --no-cache-dir .[dev]
