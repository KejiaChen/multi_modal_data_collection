#!/usr/bin/env python3
"""Upload a local recordings folder to the Hugging Face Hub.

Examples:
    python scripts/upload_recordings_to_hf.py /path/to/dataset
    python scripts/upload_recordings_to_hf.py /path/to/dataset --repo-id user/my_dataset
    python scripts/upload_recordings_to_hf.py /path/to/dataset --private

Default behavior:
    - repo_type is "dataset"
    - if --repo-id is omitted, the script derives it as:
      <logged-in-hf-username>/<sanitized-folder-name>
    - the script creates the repo if it does not already exist
    - the script uploads every file under the given folder
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi


def sanitize_repo_name(name: str) -> str:
    """Convert a folder name into a Hugging Face friendly repo name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-.")
    if not cleaned:
        raise ValueError("Could not derive a valid repo name from the folder path")
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local recordings folder to Hugging Face as a dataset repo."
    )
    parser.add_argument(
        "folder",
        help="Local folder to upload. The full directory tree will be uploaded.",
    )
    parser.add_argument(
        "--repo-id",
        help="Target Hugging Face repo id, like 'username/dataset_name'. "
        "If omitted, the script uses <your-username>/<folder-name>.",
    )
    parser.add_argument(
        "--repo-owner",
        help="Optional repo owner when deriving the repo id, for example a username or org.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=".",
        help="Destination path inside the repo. Defaults to the repo root.",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN or local hf login.",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist yet.",
    )
    visibility.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public if it does not exist yet.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Revision or branch to upload to. Defaults to 'main'.",
    )
    parser.add_argument(
        "--commit-message",
        help="Custom commit message. Defaults to 'Upload <folder-name>'.",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Glob pattern to ignore. Can be passed multiple times.",
    )
    return parser.parse_args()


def resolve_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def resolve_folder(folder: str) -> Path:
    root = Path(folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")
    return root


def resolve_repo_id(api: HfApi, folder: Path, repo_id: str | None, repo_owner: str | None) -> str:
    if repo_id:
        return repo_id
    owner = repo_owner
    if not owner:
        whoami = api.whoami()
        owner = str(whoami.get("name", "")).strip()
    if not owner:
        raise RuntimeError(
            "Could not determine the Hugging Face username. Pass --repo-id or --repo-owner explicitly."
        )
    return f"{owner}/{sanitize_repo_name(folder.name)}"


def main() -> int:
    args = parse_args()

    try:
        folder = resolve_folder(args.folder)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    token = resolve_token(args.token)
    api = HfApi(token=token)

    try:
        repo_id = resolve_repo_id(api, folder, args.repo_id, args.repo_owner)
    except Exception as exc:
        print(f"Error resolving repo id: {exc}", file=sys.stderr)
        return 2

    private = True if args.private else False if args.public else None
    commit_message = args.commit_message or f"Upload {folder.name}"

    print(f"Local folder : {folder}")
    print(f"Target repo  : {repo_id}")
    print(f"Repo type    : dataset")
    print(f"Revision     : {args.revision}")
    print(f"Repo path    : {args.path_in_repo}")

    if args.ignore:
        print(f"Ignore globs : {args.ignore}")

    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        print(f"Repo ready   : {repo_url}")
        commit_info = api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=args.path_in_repo,
            revision=args.revision,
            commit_message=commit_message,
            ignore_patterns=args.ignore or None,
        )
    except Exception as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        return 1

    print(f"Upload done  : {commit_info.oid}")
    print(f"Dataset URL  : https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
