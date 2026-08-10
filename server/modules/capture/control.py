import inspect
import json
from collections.abc import Callable

INFO_REQUEST_KEY = "info-request"
INFO_RESPONSE_KEY = "info-response"
INFO_ERROR_KEY = "info-error"
TEXT_MESSAGE_SEPARATOR = ":"


class ControlMessageRouter:
    def __init__(
        self,
        info_provider: Callable | None = None,
        event_callback=None,
        text_response_sender: Callable | None = None,
        background_info_handler: Callable | None = None,
    ):
        self.info_provider = info_provider
        self.event_callback = event_callback
        self.text_response_sender = text_response_sender
        self.background_info_handler = background_info_handler

    async def handle_text_message(self, ws, message: str) -> bool:
        payload = _decode_control_message(message)
        if payload is None or INFO_REQUEST_KEY not in payload:
            return False

        request = payload[INFO_REQUEST_KEY]
        self._emit_event("connected", _format_request_for_log(request))
        print(f"info-request received: {request!r}")

        if self.background_info_handler is not None:
            self.background_info_handler(request)
            print(f"info-request queued for analysis: {request!r}")
            return True

        if self.info_provider is None:
            response = _encode_control_message(
                INFO_ERROR_KEY,
                {
                    "code": "info_provider_missing",
                    "message": "No info provider is configured on the server.",
                    "request": request,
                },
            )
            print(f"info-request failed: no provider for {request!r}")
            await ws.send(response)
            print(f"info-error sent: {response}")
            return True

        try:
            data = self.info_provider(request)
            if inspect.isawaitable(data):
                data = await data
        except Exception as err:
            response = _encode_control_message(
                INFO_ERROR_KEY,
                {
                    "code": "info_request_failed",
                    "message": str(err),
                    "request": request,
                },
            )
            print(f"info-request failed: {err}")
            await ws.send(response)
            print(f"info-error sent: {response}")
            return True

        if isinstance(data, str):
            response = encode_plain_info_response(data)
            print(f"info-response generated: {data!r}")
            if self.text_response_sender is not None:
                sent = self.text_response_sender(data)
                if inspect.isawaitable(sent):
                    await sent
            else:
                await ws.send(response)
            print(f"info-response sent: {response}")
            return True

        response = _encode_control_message(
            INFO_RESPONSE_KEY,
            {
                "request": request,
                "data": data,
            },
        )
        print(f"info-response generated for request {request!r}: {data!r}")
        await ws.send(response)
        print(f"info-response sent: {response}")
        return True

    def _emit_event(self, state: str, message: str):
        if self.event_callback is not None:
            self.event_callback(state, message)


def _decode_control_message(message: str) -> dict | None:
    normalized_message = message.strip()
    if normalized_message == INFO_REQUEST_KEY:
        return {INFO_REQUEST_KEY: "all"}

    prefix = f"{INFO_REQUEST_KEY}:"
    if normalized_message.startswith(prefix):
        request = normalized_message[len(prefix) :].strip()
        return {INFO_REQUEST_KEY: request or "all"}

    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload

    if isinstance(payload, str) and payload.strip() == INFO_REQUEST_KEY:
        return {INFO_REQUEST_KEY: "all"}

    return None


def _encode_control_message(key: str, value) -> str:
    return json.dumps(
        {
            "type": key,
            key: value,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def encode_plain_info_response(message: str) -> str:
    return f"{INFO_RESPONSE_KEY}{TEXT_MESSAGE_SEPARATOR}{message}"


def _format_request_for_log(request) -> str:
    if isinstance(request, str):
        return f"Info request: {request}"

    if isinstance(request, list):
        return "Info request: " + ", ".join(str(item) for item in request)

    if isinstance(request, dict):
        for key in ("message", "text", "topic", "type", "name"):
            value = request.get(key)
            if value:
                return f"Info request: {value}"
        topics = request.get("topics")
        if topics:
            return f"Info request: {topics}"

    return "Info request received"
