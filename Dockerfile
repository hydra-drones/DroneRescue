FROM python:3.11.12-slim

WORKDIR /src

COPY pyproject.toml poetry.lock ./
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-root

COPY src/annotation_app ./src/annotation_app
COPY .streamlit/ .streamlit/
RUN mkdir datasamples

EXPOSE 8501
CMD ["poetry", "run", "sh", "-c", "PYTHONPATH=. streamlit run src/annotation_app/app.py --server.port=8501 --server.address=0.0.0.0"]
