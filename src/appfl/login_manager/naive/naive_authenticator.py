from appfl.login_manager import BaseAuthenticator

class NaiveAuthenticator(BaseAuthenticator):
    """
    A naive authenticator that uses a hardcoded token for authentication.
    It is only used for demonstration purposes.
    """
    def get_auth_token(self) -> dict:
        return {
            "auth_token": "appfl-naive-auth-token"
        }
    
    def validate_auth_token(self, token: dict) -> bool:
        return token.get("auth_token") == "appfl-naive-auth-token"