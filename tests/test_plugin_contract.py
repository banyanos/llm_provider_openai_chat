# -*- coding: utf-8 -*-
"""OpenAI Chat Completions plugin-contract tests (plan §6.2 / A.4)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
from llm_provider_base import LlmProviderHandler

from llm_provider_openai_chat import OpenAIChatProvider


def _agent(**config_overrides):
    configuration = {"model": "glm-5.3-flash"}
    configuration.update(config_overrides)
    return {
        "instructions": "You are terse.",
        "configuration": configuration,
        "llm": {"llm_name": "gpt"},
        "tool_call_role": "tool",
    }


def _ctx(**overrides):
    defaults = {
        "http_client": httpx.Client(),
        "credentials": {"openai_api_key": "sk-oai", "api_key": "sk-oai"},
        "endpoint": "https://api.openai.com/v1",
        "model_identifier": "glm-5.3-flash",
        "options": {},
        "trace_id": "tid-1",
        "emit": None,
        "is_cancelled": None,
        "pool_name": "httpx_llm_openai",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _handler(ctx=None, agent=None):
    return OpenAIChatProvider(
        logging.getLogger("kit"), agent or _agent(), ctx=ctx or _ctx()
    )


def _usage(**overrides):
    defaults = {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _response(tool_calls=(), content="", usage=None, response_id="resp_1"):
    message = SimpleNamespace(
        role="assistant",
        content=content or None,
        tool_calls=tool_calls or None,
    )
    return SimpleNamespace(
        id=response_id,
        usage=usage or _usage(),
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
    )


def _tool_call(call_id="call_1", name="find_order", arguments='{"q": "x"}'):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


# ---------------------------------------------------------------- metadata


def test_handler_is_plugin_base_subclass():
    assert issubclass(OpenAIChatProvider, LlmProviderHandler)


def test_plugin_metadata():
    assert OpenAIChatProvider.api_protocol == "openai-completions"
    assert OpenAIChatProvider.default_pool_name == "httpx_llm_openai"
    md = OpenAIChatProvider.plugin_metadata()
    assert isinstance(md["config_schema"], dict)
    props = md["config_schema"].get("properties", {})
    # Q8/CREDS + A.4 removals：凭据与已迁移键不入 schema
    assert "openai_api_key" not in props
    assert "max_tool_call_depth" not in props
    assert "max_retries" not in props
    assert "openai_api_key" not in md["config_schema"].get("required", [])


def test_init_requires_pooled_client():
    ctx = _ctx()
    ctx.http_client = None
    with pytest.raises(Exception, match="ctx.http_client"):
        OpenAIChatProvider(logging.getLogger("k"), _agent(), ctx=ctx)


def test_init_excludes_credential_from_model_setting():
    handler = _handler(agent=_agent(openai_api_key="leaky"))
    assert "openai_api_key" not in handler.model_setting


def test_tool_call_depth_exceeded_removed():
    # plan A.4 retires the in-handler depth budget; the class/except
    # arm remain as unreachable legacy (documented deviation, P6
    # cleanup candidate). The real check: ask_model is single-pass.
    import inspect
    src = inspect.getsource(OpenAIChatProvider.ask_model)
    assert "while True" not in src


# ------------------------------------------------- neutral → native (§5.5)


def test_to_native_passthrough_with_tool_shapes():
    handler = _handler()
    native = handler.to_native_messages(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "find_order", "arguments": '{"q": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
    )
    assert native[0]["role"] == "system"
    assert native[2]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "find_order", "arguments": '{"q": 1}'},
        }
    ]
    assert native[3] == {"role": "tool", "tool_call_id": "call_1", "content": "result"}


def test_to_native_strips_provider_state():
    handler = _handler()
    native = handler.to_native_messages(
        [
            {
                "role": "assistant",
                "content": "x",
                "provider_state": {"api_protocol": "gemini", "blocks": [1]},
            }
        ]
    )
    assert "provider_state" not in native[0]


# ------------------------------------------- single-turn handle_response


def test_handle_response_tool_branch_single_turn():
    handler = _handler()
    response = _response(
        tool_calls=[_tool_call("call_1", "find_order", '{"q": "x"}')],
        content="checking",
        usage=_usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    handler.handle_response(response, [])

    calls = handler.final_output["tool_calls"]
    assert calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "find_order", "arguments": '{"q": "x"}'},
        }
    ]
    assert handler.final_output["finish_reason"] == "tool_calls"
    assert handler.final_output["usage"]["total_tokens"] == 15


def test_handle_response_plain_answer():
    handler = _handler()
    handler.handle_response(
        _response(content="answer", usage=_usage(prompt_tokens=2,
                                                 completion_tokens=3,
                                                 total_tokens=5)),
        [],
    )
    assert handler.final_output["content"] == "answer"
    assert handler.final_output["usage"]["total_tokens"] == 5


def test_handle_response_empty_retries():
    import unittest.mock as mock

    handler = _handler()
    responses = [
        _response(content=None),
        _response(content="finally"),
    ]

    def fake_invoke(**kwargs):
        return responses.pop(0)

    with mock.patch.object(handler, "invoke_model", side_effect=fake_invoke):
        handler.handle_response(_response(content=None), [])
    assert handler.final_output["content"] == "finally"


# --------------------------------------------- single-turn handle_stream


def _emit_capture(handler):
    emitted = []
    handler.ctx = SimpleNamespace(
        **{**handler.ctx.__dict__, "emit": lambda c: emitted.append(dict(c))}
    )
    return emitted


def test_handle_stream_tool_branch_single_turn():
    handler = _handler()
    emitted = _emit_capture(handler)
    handler.run = {"run_uuid": "run-1", "thread_uuid": "thread-1"}

    class FakeAcc(dict):
        pass

    import threading as _threading
    stream_event = _threading.Event()
    def fake_stream():
        yield SimpleNamespace(
            id="resp_s1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="checking",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_5",
                                type="function",
                                function=SimpleNamespace(
                                    name="lookup", arguments='{"q": 2}'
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ]
        )
        yield SimpleNamespace(
            id="resp_s1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None),
                    finish_reason="tool_calls",
                )
            ],
        )
        yield SimpleNamespace(
            id="resp_s1",
            choices=[],
            usage=_usage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
        )

    handler.handle_stream(fake_stream(), input_messages=[], stream_event=stream_event)

    assert stream_event.is_set()
    calls = handler.final_output["tool_calls"]
    assert calls[0]["id"] == "call_5"
    assert calls[0]["function"]["name"] == "lookup"
    assert handler.final_output["finish_reason"] == "tool_calls"
    assert handler.final_output["usage"]["total_tokens"] == 9
    end_markers = [c for c in emitted if c.get("is_message_end")]
    assert end_markers, f"no message-end marker in {emitted!r}"


def test_handle_stream_plain_answer_usage():
    handler = _handler()
    _emit_capture(handler)
    handler.run = {"run_uuid": "run-1", "thread_uuid": "thread-1"}

    def fake_stream():
        yield SimpleNamespace(
            id="resp_s1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="hel", tool_calls=None),
                    finish_reason=None,
                )
            ]
        )
        yield SimpleNamespace(
            id="resp_s1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="lo", tool_calls=None),
                    finish_reason="stop",
                )
            ]
        )
        yield SimpleNamespace(
            id="resp_s1",
            choices=[],
            usage=_usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )

    handler.handle_stream(fake_stream(), input_messages=[], stream_event=None)
    assert handler.final_output["content"] == "hello"
    assert handler.final_output["usage"]["total_tokens"] == 5


# ---------------------------------------------------------------------------
# Azure OpenAI entry point (plan D8 — replaces the AZURE_OPENAI builtin)
# ---------------------------------------------------------------------------


def test_azure_metadata_replaces_builtin():
    from llm_provider_openai_chat import AzureOpenAIChatProvider

    md = AzureOpenAIChatProvider.plugin_metadata()
    assert md["api_protocol"] == "azure-openai"
    assert md["replaces_protocol"] == "AZURE_OPENAI"
    assert md["pool_name"] == "httpx_llm_azure"
    assert OpenAIChatProvider.plugin_metadata()["replaces_protocol"] == (
        "OPENAI_COMPATIBLE"
    )


def test_azure_client_uses_pooled_client_api_key_header_and_deployment():
    from llm_provider_openai_chat import AzureOpenAIChatProvider

    seen = {}

    def _respond(request):
        seen["url"] = str(request.url)
        seen["api_key"] = request.headers.get("api-key")
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(_respond))
    handler = AzureOpenAIChatProvider(
        logging.getLogger("kit"),
        _agent(
            model="gpt-4o",
            base_url="https://res.openai.azure.com/openai/deployments/x",
            api_version="2024-10-21",
            deployment="prod-4o",
        ),
        ctx=_ctx(http_client=client, credentials={"api_key": "az-key"}),
    )
    assert "api_version" not in handler.model_setting
    assert "deployment" not in handler.model_setting
    handler.client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
    )
    assert seen["url"].startswith(
        "https://res.openai.azure.com/openai/deployments/prod-4o/chat/completions"
    )
    assert "api-version=2024-10-21" in seen["url"]
    assert seen["api_key"] == "az-key"


def test_azure_requires_api_version():
    from llm_provider_openai_chat import AzureOpenAIChatProvider

    with pytest.raises(Exception):
        AzureOpenAIChatProvider(
            logging.getLogger("kit"),
            _agent(base_url="https://res.openai.azure.com"),
            ctx=_ctx(),
        )


def test_list_models_openai_compatible_and_azure_opt_out():
    from llm_provider_openai_chat import AzureOpenAIChatProvider, OpenAIChatProvider

    seen = {}

    def _respond(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"id": "glm-5.3-flash"}, {"id": "kimi-k3"}]})

    ctx = SimpleNamespace(
        http_client=httpx.Client(transport=httpx.MockTransport(_respond)),
        endpoint="https://ollama.com/v1", credentials={"api_key": "k1"}, timeout_seconds=5,
    )
    models = OpenAIChatProvider.list_models(ctx)
    assert [m["model_id"] for m in models] == ["glm-5.3-flash", "kimi-k3"]
    assert seen == {"url": "https://ollama.com/v1/models", "auth": "Bearer k1"}
    assert AzureOpenAIChatProvider.list_models(ctx) is None
    assert AzureOpenAIChatProvider.capabilities["list_models"] is False
