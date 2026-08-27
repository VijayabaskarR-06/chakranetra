# Chakranetra — Hugging Face Spaces (Docker SDK) image.
#
# Differs from the repo-root Dockerfile in three ways that Spaces requires:
#   1. runs as uid 1000 (Spaces refuses root-owned writes)
#   2. listens on 7860, the only port Spaces proxies
#   3. bakes both the model weights AND a seeded database at BUILD time, so a
#      cold container serves a populated dashboard instantly instead of running
#      a minute of CPU inference on first request.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Cache the pothole weights into the image (~23 MB) rather than downloading
# them on the first scan request.
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('keremberke/yolov8s-pothole-segmentation','best.pt')"

# Seed the ticket database by running the real pipeline over the sample trip,
# so the Space opens with the same 7 tickets the README shows.
ENV ROADLENS_DB_PATH=/home/user/app/chakranetra.db
RUN python run_demo.py && test -s chakranetra.db

EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
