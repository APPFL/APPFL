import abc


class BaseAuthenticator:
    @abc.abstractmethod
    def get_auth_token(self) -> dict[str, str]:
        """Obtain authentication token(s) in a python `dict` format with key-value pairs of `str` type."""

    @abc.abstractmethod
    def validate_auth_token(self, token: dict) -> bool:
        """Validate the authentication token. Return `True` if the token is valid, `False` otherwise."""
