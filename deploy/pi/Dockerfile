# The Pi gateway: torch-free, arm64, a few MB of Python. Built on the Pi itself
# (scripts/deploy_pi.sh), so no cross-building.
FROM python:3.12-slim
WORKDIR /app
COPY deploy/pi/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# Only what the gateway imports: the gateway package and the few pure modules of
# the chat bridge it shares addressing and parsing with. tests/gateway/
# test_torch_free.py keeps that set honest.
COPY src/gateway src/gateway
COPY src/chat/whatsapp_adapter.py src/chat/stt.py src/chat/waha_client.py \
     src/chat/documents.py src/chat/feedback.py src/chat/memory.py src/chat/response_utils.py \
     src/chat/scope.py src/chat/
COPY src/chat/static/landing.html src/chat/static/landing.html
COPY src/data/message_log.py src/data/
COPY config.yaml config.yaml
ENV PYTHONUNBUFFERED=1 GATEWAY_CONFIG=/app/config.yaml GATEWAY_LANDING_HTML=/app/src/chat/static/landing.html
CMD ["python", "-m", "src.gateway.app"]
