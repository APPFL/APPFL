"""
Version compatibility between gRPC clients and servers running different APPFL versions.

Each side sends its APPFL version in the `appfl-version` header of every RPC: a client with each
request, and the server with each response. Headers are ignored by peers that do not read them, so
they reach neither the metadata nor the configurations of older peers. A side that sends no version
runs an APPFL version older than `VERSIONED_SINCE`.

To add a mismatch rule, append a `CompatibilityRule` to `CLIENT_RULES` (checked by the server for
each client) or `SERVER_RULES` (checked by a client for the server).
"""

import logging
import threading
from collections import namedtuple
from collections.abc import Callable
from typing import Any, NamedTuple

import grpc
from packaging.version import InvalidVersion, Version

from appfl import __version__ as APPFL_VERSION

APPFL_VERSION_HEADER = "appfl-version"
VERSIONED_SINCE = "1.12.0"


class CompatibilityRule(NamedTuple):
    since: str  # the first APPFL version the peer needs
    applies: Callable[
        [Any], bool
    ]  # whether the rule applies, given this side's communicator
    problem: str  # what goes wrong with an older peer, completing "which ..."
    level: str = "warning"  # level of the message to log if the rule applies: "error", "warning" or "info"


# Rules the server checks against the version of each client
CLIENT_RULES: list[CompatibilityRule] = [
    CompatibilityRule(
        "1.12.0",
        lambda server: server.use_polling,
        "does not support the polling mode (`use_polling: True`) of this server",
        level="error",
    ),
    CompatibilityRule(
        "1.12.0",
        lambda server: True,
        "does not tell the server its `max_message_size` in the request, so the server sends messages up to its "
        "own `max_message_size`, which fail with `RESOURCE_EXHAUSTED` if the client's is smaller",
        level="warning",
    ),
]

# Rules a client checks against the version of the server
SERVER_RULES: list[CompatibilityRule] = [
    CompatibilityRule(
        "1.12.0",
        lambda client: True,
        "requires the `max_message_size` of every client to match the server's, as messages "
        "over the size limit of either side fail with `RESOURCE_EXHAUSTED`",
        level="warning",
    ),
]


class IncompatibleVersionError(Exception):
    """The APPFL version of the peer is too old for this side, so requests with it always fail."""


def _release(version: str) -> tuple[int, ...]:
    """Release numbers of a version, so that pre-releases such as 1.12.0.dev0 count as 1.12.0."""
    return Version(version).release


def check_peer_version(
    peer: str,
    peer_version: str | None,
    rules: list[CompatibilityRule],
    communicator: Any,
) -> list[tuple[str, str]]:
    """
    Check the APPFL version of a peer against the rules and this side's version.
    :param peer: Name of the peer in messages, e.g. "client 1" or "the server"
    :param peer_version: APPFL version the peer sent, or `None` if it sent none
    :param rules: `CLIENT_RULES` or `SERVER_RULES`
    :param communicator: This side's communicator, passed to `CompatibilityRule.applies`
    :return: (logging level, message) for each problem or recommendation found, where the level
        is "error", "warning" or "info", and an "error" means requests with the peer always fail
    """
    try:
        theirs = _release(peer_version) if peer_version else None
    except InvalidVersion:
        return [("warning", f"{peer} sent an invalid APPFL version {peer_version!r}.")]
    described = (
        f"APPFL {peer_version}"
        if peer_version
        else f"an APPFL version older than {VERSIONED_SINCE}"
    )
    Peer = peer[:1].upper() + peer[1:]

    messages = []
    for rule in rules:
        if theirs is not None and theirs >= _release(rule.since):
            continue
        if not rule.applies(communicator):
            continue
        if rule.level == "error":
            messages.append(
                (
                    "error",
                    f"{Peer} runs {described}, which {rule.problem}, so its requests always fail. "
                    f"Please upgrade APPFL on {peer} to {rule.since} or later.",
                )
            )
        else:
            messages.append(
                (
                    "warning",
                    f"{Peer} runs {described}, which {rule.problem}. If requests fail, this is "
                    f"a likely cause, and upgrading APPFL on {peer} to {rule.since} or later "
                    f"fixes it.",
                )
            )
    if messages:
        return messages

    # No known incompatibility, but recommend the newer minor version to the older side
    ours = _release(APPFL_VERSION)
    if theirs is None or theirs[:2] < ours[:2]:
        return [
            (
                "info",
                f"{Peer} runs {described}, older than APPFL {APPFL_VERSION} here. "
                f"Upgrading APPFL on {peer} is recommended.",
            )
        ]
    if theirs[:2] > ours[:2]:
        return [
            (
                "info",
                f"{Peer} runs {described}, newer than APPFL {APPFL_VERSION} here. "
                f"Upgrading APPFL here is recommended.",
            )
        ]
    return []


class _ClientCallDetails(
    namedtuple(
        "_ClientCallDetails",
        (
            "method",
            "timeout",
            "metadata",
            "credentials",
            "wait_for_ready",
            "compression",
        ),
    ),
    grpc.ClientCallDetails,
):
    pass


class VersionCheckInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """
    Client interceptor that sends the APPFL version of the client with every RPC, and checks the
    version of the server against `SERVER_RULES` in the response headers of the first RPC the
    server answers. Problems are logged once. If the server is incompatible, that RPC and every
    later one raise `IncompatibleVersionError`.
    """

    def __init__(self, communicator: Any):
        self._communicator = communicator
        self._lock = threading.Lock()
        self._checked = False
        self._error = (
            None  # message of `IncompatibleVersionError` if the server is incompatible
        )

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return self._intercept(continuation, client_call_details, request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return self._intercept(continuation, client_call_details, request)

    def intercept_stream_stream(
        self, continuation, client_call_details, request_iterator
    ):
        return self._intercept(continuation, client_call_details, request_iterator)

    def _intercept(self, continuation, client_call_details, request):
        if self._error is not None:
            raise IncompatibleVersionError(self._error)
        metadata = list(client_call_details.metadata or [])
        metadata.append((APPFL_VERSION_HEADER, APPFL_VERSION))
        call = continuation(
            _ClientCallDetails(
                client_call_details.method,
                client_call_details.timeout,
                metadata,
                client_call_details.credentials,
                client_call_details.wait_for_ready,
                client_call_details.compression,
            ),
            request,
        )
        if not self._checked:
            self._check_server(call)
        return call

    def _check_server(self, call):
        # Blocks until the server answers the RPC with its headers, or the RPC fails
        server_metadata = call.initial_metadata()
        if call.done() and call.code() != grpc.StatusCode.OK:
            # Failed before the server could answer, so check the server on a later RPC
            return
        with self._lock:
            if self._checked:
                return
            server_version = dict(server_metadata or ()).get(APPFL_VERSION_HEADER)
            logger = self._communicator.logger or logging.getLogger(__name__)
            errors = []
            for level, message in check_peer_version(
                "the server", server_version, SERVER_RULES, self._communicator
            ):
                getattr(logger, level)(message)
                if level == "error":
                    errors.append(message)
            if errors:
                self._error = " ".join(errors)
            self._checked = True
        if self._error is not None:
            call.cancel()
            raise IncompatibleVersionError(self._error)
