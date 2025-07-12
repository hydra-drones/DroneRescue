from __future__ import annotations

import os
import tempfile
from pathlib import Path

import typer
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from tqdm import tqdm
from loguru import logger

SCOPES = ["https://www.googleapis.com/auth/drive"]


__all__ = [
    "get_service",
    "download_folder_to_temp",
]


def get_service(credentials_file: str, token_file: str) -> Resource:
    """Authorize and build Drive v3 service.

    Args:
        credentials_file (str): path to .json file with credentials
        token_file (str): path to .json file with token

    Returns:
        Resource: object for interacting with an Google Drive API
    """
    creds: Credentials | None = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, "w", encoding="utf-8") as tf:
            tf.write(creds.to_json())

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_files_in_folder(service: Resource, folder_id: str) -> list[dict]:
    """List all files in the given folder on Google Drive

    Args:
        service: object for interacting with an Google Drive API
        folder_id: the folder signature, can be found inside the URL of the folder

    Returns:
        list[dict]: #TODO
    """
    query = f"'{folder_id}' in parents and trashed = false"
    files: list[dict] = []
    page_token: str | None = None
    while True:
        resp = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id,name,mimeType,size,modifiedTime)",
                pageSize=1000,
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if page_token is None:
            break
    return files


def download_folder(service: Resource, folder_id: str, destination: Path) -> Path:
    """Download given folder from Google Drive to the local distination

    Args:
        service: object for interacting with an Google Drive API
        folder_id: the folder signature, can be found inside the URL of the folder
        destination: path to local folder

    Returns:
        Path: path to folder where the files were saved
    """
    destination.mkdir(parents=True, exist_ok=True)
    files = list_files_in_folder(service, folder_id)
    if not files:
        logger.warning("Folder is empty. Nothing to download.")
        return
    logger.info(f"Found {len(files)} items. Starting download.")
    for meta in tqdm(files, unit="file"):
        fpath = destination / meta["name"]
        request = service.files().get_media(fileId=meta["id"])
        with fpath.open("wb") as fh:
            dl = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = dl.next_chunk()
    logger.info(f"Download finished --> {destination.resolve()}")

    return destination.resolve()


def upload_file(
    service: Resource, local_path: Path, parent_folder_id: str | None
) -> None:
    """Upload local file into Google Drive

    Args:
        service: object for interacting with an Google Drive API
        local_path: path to local file
        parent_folder_id: ID of the Google Drive's folder where the file should be uploaded.
            If :code:`None`, file will be aploaded to :code:`root` Google Drive folder

    Raises:
        FileNotFoundError: local file not found
    """
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    meta = {"name": local_path.name}
    if parent_folder_id:
        meta["parents"] = [parent_folder_id]
    media = MediaFileUpload(local_path, resumable=True)
    req = service.files().create(body=meta, media_body=media, fields="id")
    resp = None
    with tqdm(
        total=100, unit="%", leave=False, desc=f"Uploading {local_path.name}"
    ) as bar:
        while resp is None:
            status, resp = req.next_chunk()
            if status:
                bar.update(int(status.progress() * 100) - bar.n)
    logger.info(f"Uploaded --> https://drive.google.com/file/d/{resp['id']}/view")


def download_folder_to_temp(
    folder_id: str = typer.Argument(
        ..., help="Google Drive folder's ID (you can find it in the URL)"
    ),
    credentials_file: Path = typer.Option(
        "credentials.json",
        "--credentials-file",
        "-c",
        help="OAuth client secrets (JSON file)",
    ),
    token_file: Path = typer.Option(
        "token.json",
        "--token-file",
        "-t",
        help="Token has been generated after first login. (JSON File)",
    ),
):
    """Download given folder from Google Drive and save it to temporary local folder

    After using it requires to delete the temporary folder with :code:`shutil.rmtree`

    Args:
        folder_id (str): Google Drive folder's ID (you can find it in the URL).
        credentials_file (Path | None): OAuth client secrets (JSON file). If None, default path is "./credentials.json"
        token_file (Path | None): Token has been generated after first login. (JSON File). If None, default path is "./token.json"

    Returns:
        Path: path to temporary folder where the files were saved
    """
    service = get_service(str(credentials_file), str(token_file))
    tmp_dir = Path(tempfile.mkdtemp(prefix="drone_rescue_"))
    return download_folder(service, folder_id, tmp_dir)


def download(
    folder_id: str = typer.Argument(
        ..., help="Google Drive folder's ID (you can find it in the URL)"
    ),
    destination: Path = typer.Option(
        "downloads",
        "--destination",
        "-d",
        help="Path to local dir where the folder will be saved",
    ),
    credentials_file: Path = typer.Option(
        "credentials.json",
        "--credentials-file",
        "-c",
        help="OAuth client secrets (JSON file)",
    ),
    token_file: Path = typer.Option(
        "token.json",
        "--token-file",
        "-t",
        help="Token has been generated after first login. (JSON File)",
    ),
):
    """Downlaod folder from Google Drive into local dir

    Args:
        folder_id (str): Google Drive folder's ID (you can find it in the URL).
        destination (Path | None): Path to local dir where the folder will be saved. If None, folder will be saved to the default "downloads" folder
        credentials_file (Path | None): OAuth client secrets (JSON file). If None, default path is "./credentials.json"
        token_file (Path | None): Token has been generated after first login. (JSON File). If None, default path is "./token.json"
    """
    service = get_service(str(credentials_file), str(token_file))
    try:
        download_folder(service, folder_id, destination)
    except Exception as e:
        logger.error(
            f"Cannot download folder with ID: {folder_id} into {destination}. {e}"
        )


def upload(
    local_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to file to be saved",
    ),
    parent: str
    | None = typer.Option(
        None,
        "--parent",
        "-p",
        help="Folder's ID on Google Drive where the file will be saved",
    ),
    credentials_file: Path = typer.Option(
        "credentials.json",
        "--credentials-file",
        "-c",
        help="OAuth client secrets (JSON file)",
    ),
    token_file: Path = typer.Option(
        "token.json",
        "--token-file",
        "-t",
        help="Token has been generated after first login. (JSON File)",
    ),
):
    """Upload file from local to Google Drive

    Args:
        local_path (Path): Path to file to be saved
        parent (str | None): Google Drive folder's ID. If None, will be saved to root
        credentials_file (Path | None): OAuth client secrets (JSON file). If None, default path is "./credentials.json"
        token_file (Path | None): Token has been generated after first login. (JSON File). If None, default path is "./token.json"
    """
    service = get_service(str(credentials_file), str(token_file))
    try:
        upload_file(service, local_path, parent)
    except Exception as e:
        logger.error(f"Cannot upload folder {local_path} into {parent}. {e}")
