# FROM python:3.10.6
# WORKDIR /src/app
# RUN mkdir log
# COPY ./requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . ./
# ENTRYPOINT gunicorn app:app --workers 1 --worker-class app.ConfigurableWorker --bind 0.0.0.0:5080 --log-level debug

################ build
FROM python:3.10.6-slim as builder

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

# install poetry
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -sSL https://install.python-poetry.org | python3 - --version 1.5.1

# Add `poetry` to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install dependencies
WORKDIR /src/app
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-cache --no-root


######################## Run
FROM python:3.10.6-slim 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/src/app/.venv/bin:$PATH"

WORKDIR /src/app
# Run your app
COPY --from=builder /src/app/.venv ./.venv
COPY . /src/app
RUN python -c "from main import *"
CMD gunicorn main:app --workers ${NUM_WORKERS:-1} --worker-class uvicorn.workers.UvicornWorker \
                 --backlog ${REQUEST_QUEUE_SIZE:-NUM_WORKERS} \
                --bind 0.0.0.0:8000 --log-level debug --error-logfile log/server.log

                # 