FROM python:3.10

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

# For Flask, we should use gunicorn or similar, not uvicorn (which is for FastAPI)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]