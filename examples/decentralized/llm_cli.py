"""Shared LLM command-line flags, so every runner exposes the same switches.

Precedence is flags > environment > defaults, and the API key is read from the environment
by default so it never has to appear in a shell history or a checked-in config.

    APPFL_DECENTRALIZED_LLM_ENABLED / _BASE_URL / _API_KEY / _MODEL / _ALLOW_REMOTE / _CACHE
    OPENAI_API_KEY, OPENAI_BASE_URL      (fallbacks)
"""

from __future__ import annotations

import argparse

from appfl.decentralized.algorithm.adko import LLMConfig


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("LLM (ADKO Algorithm 1 steps 3 and 10)")
    group.add_argument(
        "--llm", action="store_true",
        help="enable the language model; omit for the LM-free ablation arm",
    )
    group.add_argument(
        "--llm-base-url", default=None,
        help="OpenAI-compatible endpoint (vLLM, Argo, gateway). Omit for the OpenAI API.",
    )
    group.add_argument(
        "--llm-api-key", default=None,
        help="bearer token; defaults to $APPFL_DECENTRALIZED_LLM_API_KEY or $OPENAI_API_KEY",
    )
    group.add_argument("--llm-model", default=None, help="model name (default gpt-4o-mini)")
    group.add_argument(
        "--llm-temperature", type=float, default=None,
        help="sampling temperature; pass --llm-no-temperature to omit it entirely",
    )
    group.add_argument(
        "--llm-no-temperature", action="store_true",
        help="do not send temperature at all (some reasoning models reject any non-default)",
    )
    group.add_argument(
        "--llm-max-tokens", type=int, default=None, help="output cap (default 512)",
    )
    group.add_argument(
        "--llm-token-param", default=None,
        choices=["max_tokens", "max_completion_tokens", "none"],
        help="which key carries the output cap; 'none' omits it and lets the server decide",
    )
    group.add_argument("--llm-timeout", type=float, default=None, help="seconds per call")
    group.add_argument(
        "--llm-cache", default=None,
        help="SQLite prompt cache path; strongly recommended for multi-seed sweeps",
    )
    group.add_argument(
        "--llm-allow-remote", action="store_true",
        help="permit sending raw observations to a non-local endpoint (see llm.py warning)",
    )
    group.add_argument(
        "--llm-no-insight", action="store_true",
        help="skip the z field; candidate proposal only, which never sees raw observations",
    )
    group.add_argument(
        "--llm-no-propose", action="store_true",
        help="skip candidate proposal; write insights only",
    )
    group.add_argument("--llm-verbose", action="store_true", help="log endpoint failures")


def llm_config_from_args(args: argparse.Namespace) -> LLMConfig:
    """Build an :class:`LLMConfig`, validating before any agent is constructed.

    Validating here rather than at first call means a misconfigured run fails in the first
    second instead of thirty rounds in.
    """
    config = LLMConfig.from_env(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
        temperature=args.llm_temperature,
        timeout_seconds=args.llm_timeout,
        cache_path=args.llm_cache,
        max_output_tokens=args.llm_max_tokens,
    )
    if args.llm_no_temperature:
        config.temperature = None
    if args.llm_token_param is not None:
        config.token_limit_param = (
            None if args.llm_token_param == "none" else args.llm_token_param
        )
    if args.llm:
        config.enabled = True
    if args.llm_allow_remote:
        config.allow_remote = True
    if args.llm_no_insight:
        config.emit_insight = False
    if args.llm_no_propose:
        config.propose = False
    if args.llm_verbose:
        config.verbose = True
    config.validate()
    return config
