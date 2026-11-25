import inspect
from globus_sdk import AccessTokenAuthorizer
from globus_sdk.scopes import AuthScopes, ComputeScopes
from globus_compute_sdk import Client


def create_token_client(
    compute_token: str, openid_token: str, **client_kwargs
) -> Client:
    """
    Create a Globus Compute client authorized by access tokens instead of an interactive login.

    Newer SDKs take the compute token through `Client(authorizer=...)`, and
    globus-compute-sdk 4.20 removed the `AuthorizerLoginManager` this used to go through.
    The SDKs available to Python 3.8 (<= 2.29) predate `authorizer`, so they still get the
    login manager, which is the only place `openid_token` is used.

    :param `compute_token`: Globus Compute access token.
    :param `openid_token`: Globus Auth (OpenID) access token; only used by SDKs without `authorizer`.
    :param `client_kwargs`: further keyword arguments for `Client`, e.g. `code_serialization_strategy`.
    """
    if "authorizer" in inspect.signature(Client).parameters:
        return Client(authorizer=AccessTokenAuthorizer(compute_token), **client_kwargs)
    from globus_compute_sdk.sdk.login_manager import AuthorizerLoginManager

    login_manager = AuthorizerLoginManager(
        authorizers={
            ComputeScopes.resource_server: AccessTokenAuthorizer(compute_token),
            AuthScopes.resource_server: AccessTokenAuthorizer(openid_token),
        }
    )
    login_manager.ensure_logged_in()
    return Client(login_manager=login_manager, **client_kwargs)
