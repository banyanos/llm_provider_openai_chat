#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

__all__ = ["AzureOpenAIChatProvider", "OpenAIChatProvider"]
from .llm_provider_openai_chat import (
    AzureOpenAIChatProvider,
    OpenAIChatProvider,
)

#: Discovered by llm_engine's ``llm_provider_*`` package scan when the
#: project is on PYTHONPATH without being pip-installed (bind mount); the
#: ``banyan.llm_providers`` entry points cover installed packages.
PROVIDERS = [OpenAIChatProvider, AzureOpenAIChatProvider]
