ARG FRAMEWORK=pytorch
ARG TAG=latest
FROM ${FRAMEWORK}/${FRAMEWORK}:${TAG} as base

SHELL ["/bin/bash", "-c"]

WORKDIR /app

COPY lm-polygraph ./lm-polygraph
RUN pip install --upgrade pip && pip install -e ./lm-polygraph

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]