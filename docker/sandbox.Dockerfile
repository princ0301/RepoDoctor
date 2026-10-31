FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    PATH="/opt/venv/bin:$PATH"

# git: some builds read it (hatch-vcs, setuptools-scm). The mounted repo is owned
# by a different user, so mark every directory as safe for git.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

RUN python -m venv /opt/venv && pip install --upgrade pip

WORKDIR /work