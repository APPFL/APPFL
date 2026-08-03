from typing import Dict, Optional
import requests
import jwt
from appfl.login_manager import BaseAuthenticator


class KeycloakAuthenticator(BaseAuthenticator):
    """
    Authenticator for FL server and client using Keycloak (OIDC) with the
    client credentials grant. Each FL client is registered in Keycloak as
    its own confidential client (`client_id` + `client_secret`) and must be
    assigned the `fl_client` role to be authorized to join the federation.

    :param is_fl_server: Whether the authenticator is for the FL server or client.
    :param server_url: Base URL of the Keycloak server, e.g. "https://keycloak.example.org".
    :param realm: Keycloak realm name.
    :param client_id: FL client's Keycloak client ID. Required for FL client.
    :param client_secret: FL client's Keycloak client secret. Required for FL client.
    :param role: Name of the role a client's token must carry to be authorized.
    :param role_client_id: If `role` is a client role rather than a realm role,
        the client ID under whose `resource_access` claim it should be looked up.
        Leave `None` to check `role` as a realm role.
    :param leeway: Clock-skew leeway (seconds) allowed when validating token expiry.
    """

    def __init__(
        self,
        *,
        is_fl_server: bool = False,
        server_url: str,
        realm: str,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        role: str = "fl_client",
        role_client_id: Optional[str] = None,
        leeway: int = 10,
    ) -> None:
        self.is_fl_server = is_fl_server
        self.realm = realm
        self.role = role
        self.role_client_id = role_client_id
        self.leeway = leeway

        issuer = f"{server_url.rstrip('/')}/realms/{realm}"
        self.issuer = issuer
        self.token_endpoint = f"{issuer}/protocol/openid-connect/token"

        if self.is_fl_server:
            # PyJWKClient caches the fetched key set, so signature verification
            # does not hit Keycloak on every `validate_auth_token()` call.
            self._jwk_client = jwt.PyJWKClient(
                f"{issuer}/protocol/openid-connect/certs"
            )
        else:
            assert client_id is not None and client_secret is not None, (
                "FL client must provide `client_id` and `client_secret` to "
                "authenticate with Keycloak."
            )
            self.client_id = client_id
            self.client_secret = client_secret

    def get_auth_token(self) -> Dict[str, str]:
        """
        Invoked by the FL client to obtain an access token from Keycloak via
        the client credentials grant.
        """
        assert not self.is_fl_server, "Server does not need to obtain auth tokens"
        response = requests.post(
            self.token_endpoint,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=10,
        )
        response.raise_for_status()
        return {"access_token": response.json()["access_token"]}

    def validate_auth_token(self, token: Dict) -> bool:
        """
        Invoked by the FL server to validate a client's access token: verifies
        the JWT's signature, issuer, and expiry against Keycloak's JWKS, then
        checks that the token carries the required role.
        """
        assert self.is_fl_server, "Client does not need to validate auth tokens"
        access_token = token.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return False
        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(access_token)
            claims = jwt.decode(
                access_token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self.issuer,
                leeway=self.leeway,
                # Keycloak's default `aud` claim is not client-specific (it is
                # typically "account"), so authorization here relies on the
                # role check below rather than audience matching.
                options={"verify_aud": False},
            )
        except jwt.PyJWTError:
            return False
        return self._has_role(claims)

    def _has_role(self, claims: Dict) -> bool:
        if self.role_client_id is not None:
            roles = (
                claims.get("resource_access", {})
                .get(self.role_client_id, {})
                .get("roles", [])
            )
        else:
            roles = claims.get("realm_access", {}).get("roles", [])
        return self.role in roles
