FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

RUN apt-get update && apt-get install -y \
    python3.12 python3-pip python3.12-venv git curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

RUN pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install "sam3 @ git+https://github.com/facebookresearch/sam3.git"

COPY app /app/app
COPY app/scripts /app/scripts

RUN chmod +x /app/scripts/start.sh

CMD ["/app/scripts/start.sh"]