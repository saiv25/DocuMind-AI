"""Create or update a PBKDF2-hashed local DocuMind login."""

from __future__ import annotations

import getpass
import hashlib
import json
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
USER_FILE = BASE_DIR / "users.json"


def main() -> None:
    username = input("Username: ").strip()

    if (
        not username
        or any(char in username for char in "\\/\n\r")
    ):
        raise SystemExit(
            "Enter a non-empty username without slashes or newlines."
        )

    password = getpass.getpass(
        "Password (minimum 12 characters): "
    )

    confirm = getpass.getpass(
        "Confirm password: "
    )

    if len(password) < 12:
        raise SystemExit(
            "Choose a password of at least 12 characters."
        )

    if password != confirm:
        raise SystemExit("Passwords do not match.")

    salt = secrets.token_bytes(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        260_000,
    ).hex()

    data = {}

    if USER_FILE.exists():
        try:
            data = json.loads(
                USER_FILE.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise SystemExit(
                f"Existing users.json could not be read: {exc}"
            )

    data[username] = {
        "salt": salt.hex(),
        "password_hash": digest,
    }

    USER_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    print(f"Saved credentials for {username!r} to {USER_FILE}.")
    print(
        "Keep users.json out of Git. Enable authentication with "
        "DOCUMIND_AUTH_ENABLED=true in .env."
    )


if __name__ == "__main__":
    main()
    