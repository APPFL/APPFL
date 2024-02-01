from __future__ import annotations
import sys
import globus_sdk
if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable, Optional
else:
    from typing_extensions import Protocol, runtime_checkable, Optional

@runtime_checkable
class GlobusLoginManagerProtocol(Protocol):
    def ensure_logged_in(self) -> bool:
        ...
    
    def logout(self) -> None:
        ...

    def get_identity_client(self) -> globus_sdk.AuthClient:
        ...

    def get_group_client(self) -> globus_sdk.GroupsClient:
        ...

    def get_auth_token(self) -> dict:
        ...

    def get_identity_client_with_tokens(self, access_token=None, refresh_token=None, expires_at=None) -> Optional[globus_sdk.AuthClient]:
        ...