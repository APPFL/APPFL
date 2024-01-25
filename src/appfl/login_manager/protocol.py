from __future__ import annotations
import sys
import globus_sdk
if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable

@runtime_checkable
class LoginManagerProtocol(Protocol):
    def ensure_logged_in(self) -> None:
        ...
    
    def logout(self) -> None:
        ...

    def get_auth_client(self) -> globus_sdk.AuthClient:
        ...

    def get_group_client(self) -> globus_sdk.GroupsClient:
        ...