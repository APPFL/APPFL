from typing import Dict
from appfl.login_manager import BaseAuthenticator


class NaiveAuthenticator(BaseAuthenticator):
    """
    A naive authenticator that uses a hardcoded token for authentication.
    It is only used for demonstration purposes.
    """

    def __init__(self, *, auth_token: str = "appfl-naive-auth-token"):
        self.auth_token = auth_token

    def get_auth_token(self) -> Dict[str, str]:
        return {
            "auth_token": self.auth_token,
        }

    def validate_auth_token(self, token: dict) -> bool:
        return token.get("auth_token") == self.auth_token
