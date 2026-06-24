"""Tests for description configuration default values and environment variable overrides."""

from __future__ import annotations

import importlib
import os
import sys
from rag_ingestion import config


def test_config_defaults():
    # Save existing env vars
    saved_input_chars = os.environ.get("CHUNK_DESCRIPTION_MAX_INPUT_CHARS")
    saved_num_ctx = os.environ.get("CODESEEK_DESCRIPTION_NUM_CTX")

    # Clear from environment to test fallback defaults
    if "CHUNK_DESCRIPTION_MAX_INPUT_CHARS" in os.environ:
        del os.environ["CHUNK_DESCRIPTION_MAX_INPUT_CHARS"]
    if "CODESEEK_DESCRIPTION_NUM_CTX" in os.environ:
        del os.environ["CODESEEK_DESCRIPTION_NUM_CTX"]

    try:
        importlib.reload(config)
        assert config.CHUNK_DESCRIPTION_MAX_INPUT_CHARS == 1800
        assert config.CODESEEK_DESCRIPTION_NUM_CTX == 4096
        # Unrelated defaults did not change
        assert config.CODESEEK_DESCRIPTION_MAX_CHARS == 600
        assert config.CODESEEK_LABEL_NUM_CTX == 2048
    finally:
        # Restore env vars
        if saved_input_chars is not None:
            os.environ["CHUNK_DESCRIPTION_MAX_INPUT_CHARS"] = saved_input_chars
        if saved_num_ctx is not None:
            os.environ["CODESEEK_DESCRIPTION_NUM_CTX"] = saved_num_ctx
        importlib.reload(config)


def test_config_env_overrides():
    # Save existing env vars
    saved_input_chars = os.environ.get("CHUNK_DESCRIPTION_MAX_INPUT_CHARS")
    saved_num_ctx = os.environ.get("CODESEEK_DESCRIPTION_NUM_CTX")

    # Set override values
    os.environ["CHUNK_DESCRIPTION_MAX_INPUT_CHARS"] = "2500"
    os.environ["CODESEEK_DESCRIPTION_NUM_CTX"] = "8192"

    try:
        importlib.reload(config)
        assert config.CHUNK_DESCRIPTION_MAX_INPUT_CHARS == 2500
        assert config.CODESEEK_DESCRIPTION_NUM_CTX == 8192
    finally:
        # Restore env vars
        if saved_input_chars is not None:
            os.environ["CHUNK_DESCRIPTION_MAX_INPUT_CHARS"] = saved_input_chars
        else:
            if "CHUNK_DESCRIPTION_MAX_INPUT_CHARS" in os.environ:
                del os.environ["CHUNK_DESCRIPTION_MAX_INPUT_CHARS"]

        if saved_num_ctx is not None:
            os.environ["CODESEEK_DESCRIPTION_NUM_CTX"] = saved_num_ctx
        else:
            if "CODESEEK_DESCRIPTION_NUM_CTX" in os.environ:
                del os.environ["CODESEEK_DESCRIPTION_NUM_CTX"]

        importlib.reload(config)
