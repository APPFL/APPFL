from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import requests
import torchvision
from torchvision.datasets.utils import check_integrity

logger = logging.getLogger(__name__)

__all__ = ["download_data"]


# NOTE:
# Some original LEAF URLs became unstable over time (e.g., Sent140 Stanford host).
# We keep official endpoints and add alternatives where possible.
RESOURCE_MAP = {
    "femnist": [
        {
            "name": "by_class",
            "filename": "by_class.zip",
            "sources": [
                "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",
            ],
            "md5": "79572b1694a8506f2b722c7be54130c4",
            "extract": True,
            "remove_archive": True,
            "ready_paths": ["by_class"],
        },
        {
            "name": "by_write",
            "filename": "by_write.zip",
            "sources": [
                "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip",
            ],
            "md5": "a29f21babf83db0bb28a2f77b2b456cb",
            "extract": True,
            "remove_archive": True,
            "ready_paths": ["by_write"],
        },
    ],
    "shakespeare": [
        {
            "name": "shakespeare_raw_text",
            "filename": "100.txt",
            "sources": [
                "https://www.gutenberg.org/ebooks/100.txt.utf-8",
                "https://www.gutenberg.org/cache/epub/100/pg100.txt",
            ],
            "extract": False,
            "ready_paths": ["100.txt"],
        },
    ],
    "sent140": [
        {
            "name": "sent140_csv",
            "filename": "trainingandtestdata.zip",
            "sources": [
                "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip",
                "https://zenodo.org/records/11116734/files/trainingandtestdata.zip?download=1",
            ],
            "md5": "1647eb110dd2492512e27b9a70d5d1bc",
            "extract": True,
            "remove_archive": True,
            "ready_paths": [
                "training.1600000.processed.noemoticon.csv",
                "testdata.manual.2009.06.14.csv",
            ],
        },
        {
            "name": "glove_6b",
            "filename": "glove.6B.zip",
            "sources": [
                "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
                "https://nlp.stanford.edu/data/glove.6B.zip",
            ],
            "md5": "056ea991adb4740ac6bf1b6d9b50408b",
            "extract": True,
            "remove_archive": True,
            "ready_paths": ["glove.6B.300d.txt"],
        },
    ],
    "celeba": [
        {
            "name": "identity_file",
            "filename": "identity_CelebA.txt",
            "sources": ["gdrive:1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS"],
            "extract": False,
            "ready_paths": ["identity_CelebA.txt"],
        },
        {
            "name": "attribute_file",
            "filename": "list_attr_celeba.txt",
            "sources": ["gdrive:0B7EVK8r0v71pblRyaVFSWGxPY0U"],
            "extract": False,
            "ready_paths": ["list_attr_celeba.txt"],
        },
        {
            "name": "images_zip",
            "filename": "img_align_celeba.zip",
            "sources": [
                "gdrive:0B7EVK8r0v71pZjFTYXZWM3FlRnM",
                "https://cseweb.ucsd.edu/~weijian/static/datasets/celeba/img_align_celeba.zip",
            ],
            "md5": "00d2c5bc6d35e252742224ab0c1e8fcb",
            "extract": True,
            "remove_archive": True,
            "ready_paths": ["img_align_celeba"],
        },
    ],
    "reddit": [
        {
            "name": "reddit_subsampled",
            "filename": "reddit_subsampled.zip",
            "sources": ["gdrive:1ISzp69JmaIJqBpQCX-JJ8-kVyUns8M7o"],
            "extract": True,
            "remove_archive": False,
            "ready_paths": ["new_small_data"],
        },
    ],
}

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response_content(path: Path, response):
    chunk_size = 32768
    with path.open("wb") as file:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                file.write(chunk)


def _resource_ready(download_root: Path, resource) -> bool:
    ready_paths = resource.get("ready_paths", [])
    if not ready_paths:
        return False
    return all((download_root / rel).exists() for rel in ready_paths)


def _download_file_from_google_drive(
    download_root: Path,
    file_name: str,
    identifier: str,
    md5: str | None = None,
):
    base_url = "https://docs.google.com/uc?export=download"
    target_file = download_root / file_name

    if target_file.exists() and (md5 is None or check_integrity(str(target_file), md5)):
        return

    session = requests.Session()
    response = session.get(
        base_url,
        params={"id": identifier, "confirm": 1},
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    token = _get_confirm_token(response)

    if token:
        params = {"id": identifier, "confirm": token}
        response = session.get(base_url, params=params, stream=True, timeout=60)
        response.raise_for_status()

    _save_response_content(target_file, response)

    if md5 is not None and not check_integrity(str(target_file), md5):
        raise RuntimeError(f"Corrupted download (md5 mismatch): {target_file}")


def _download_from_source(download_root: Path, source: str, resource):
    filename = resource["filename"]
    md5 = resource.get("md5")
    extract = bool(resource.get("extract", False))
    remove_archive = bool(resource.get("remove_archive", True))

    if source.startswith("gdrive:"):
        file_id = source.split(":", 1)[1]
        _download_file_from_google_drive(
            download_root=download_root,
            file_name=filename,
            identifier=file_id,
            md5=md5,
        )
        if extract and filename.endswith(".zip"):
            with zipfile.ZipFile(download_root / filename, "r", compression=zipfile.ZIP_STORED) as zf:
                zf.extractall(download_root)
            if remove_archive:
                try:
                    os.remove(download_root / filename)
                except OSError:
                    pass
        return

    if extract:
        torchvision.datasets.utils.download_and_extract_archive(
            url=source,
            download_root=str(download_root),
            filename=filename,
            md5=md5,
            remove_finished=remove_archive,
        )
    else:
        torchvision.datasets.utils.download_url(
            url=source,
            root=str(download_root),
            filename=filename,
            md5=md5,
        )


def _ensure_resource(download_root: Path, dataset_name: str, resource, logger_override=None):
    active_logger = logger_override or logger
    if _resource_ready(download_root, resource):
        return

    errors = []
    for source in resource.get("sources", []):
        try:
            active_logger.info(
                "[LEAF-%s] downloading `%s` from %s",
                dataset_name.upper(),
                resource.get("name", resource["filename"]),
                source,
            )
            _download_from_source(download_root, source, resource)
            if _resource_ready(download_root, resource):
                return
        except Exception as exc:
            errors.append(f"{source} -> {exc}")

    name = resource.get("name", resource["filename"])
    details = "\n".join(errors) if errors else "no source candidates configured"
    raise RuntimeError(
        f"[LEAF-{dataset_name.upper()}] failed to download `{name}`.\n{details}"
    )


def download_data(download_root, dataset_name, logger=None):
    """Download LEAF raw data with fallback mirrors for brittle/stale links."""
    active_logger = logger or globals()["logger"]
    dataset_name = str(dataset_name).strip().lower()
    if dataset_name not in RESOURCE_MAP:
        raise ValueError(
            f"Unsupported LEAF dataset `{dataset_name}`. "
            f"Supported: {sorted(RESOURCE_MAP.keys())}"
        )

    root = Path(download_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    active_logger.info("[LEAF-%s] start downloading.", dataset_name.upper())
    for resource in RESOURCE_MAP[dataset_name]:
        _ensure_resource(root, dataset_name, resource, logger_override=active_logger)
    active_logger.info("[LEAF-%s] finished downloading.", dataset_name.upper())
