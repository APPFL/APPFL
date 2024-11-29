from __future__ import annotations

import os
import pathlib
from globus_sdk.tokenstorage import SQLiteAdapter


def _home() -> pathlib.Path:
    return pathlib.Path.home()


def ensure_appfl_dir() -> pathlib.Path:
    """
    Ensure that the appfl storage directory exists and is a directory.
    """
    dirname = _home() / ".appfl" / "globus_auth"
    user_dirname = os.getenv("APPFL_USER_DIR")
    if user_dirname:
        dirname = pathlib.Path(user_dirname)
    if dirname.is_dir():
        pass
    elif dirname.is_file():
        raise FileExistsError(
            f"Error creating directory {dirname}, "
            "please rename or remove the confilicting file."
        )
    else:
        dirname.mkdir(
            mode=0o700,
            parents=True,
            exist_ok=True,
        )
    return dirname


def _get_storage_filename() -> str:
    """
    Return the path to the SQLite token storage file.
    """
    dirname = ensure_appfl_dir()
    return os.path.join(dirname, "storage.db")


def _resolve_namespace(is_fl_server: bool) -> str:
    """
    Return the namespace for saving tokens:
    `appfl_server` if invoked by an FL server, and `appfl_client` if invoked by an FL client.

    :param `is_fl_server`: True if invoked by an FL server, False if invoked by an FL client.
    """
    if is_fl_server:
        return "appfl_server"
    else:
        return "appfl_client"


def get_token_storage_adapter(*, is_fl_server: bool) -> SQLiteAdapter:
    """
    Return the SQLite token storage adapter.

    :param `is_fl_server`: True if invoked by an FL server, False if invoked by an FL client.
    """
    filename = _get_storage_filename()
    namespace = _resolve_namespace(is_fl_server)
    return SQLiteAdapter(
        filename,
        namespace=namespace,
        connect_params={"check_same_thread": False},
    )
