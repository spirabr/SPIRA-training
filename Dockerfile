LABEL org.opencontainers.image.source="https://github.com/spirabr/SPIRA-training"

# Arguments to configure Python
ARG PYTHON_VERSION=3.12

# Create build image
FROM python:${PYTHON_VERSION} AS build

ARG PIP_VERSION=23.2.1
ARG POETRY_VERSION=1.6.1

LABEL org.opencontainers.image.source="https://github.com/spirabr/SPIRA-training"

WORKDIR /app

# Set pip"s standard version
RUN pip install pip==${PIP_VERSION} poetry==${POETRY_VERSION}

COPY poetry.toml pyproject.toml poetry.lock ./

RUN poetry install

# Copy source code
COPY . .

RUN poetry install

# Run API
CMD ["poetry", "run", "python", "./src/spira_training/main.py"]