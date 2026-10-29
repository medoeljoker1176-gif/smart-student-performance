"""
make_password_hash.py - create a password hash for .streamlit/secrets.toml

Run:   python tools/make_password_hash.py teacher1 teacher "Dr. Ahmed"
It asks for the password (hidden) and prints a block to paste into .streamlit/secrets.toml.
Never put a plain-text password in the secrets file or in Git.
"""
import getpass
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.auth import MIN_PASSWORD_LENGTH, ROLES, hash_password  # noqa: E402


def main(argv):
    if len(argv) < 3 or argv[2] not in ROLES:
        sys.exit(f"Usage: python tools/make_password_hash.py <username> <{'|'.join(ROLES)}> [display name]")
    username, role = argv[1].strip().lower(), argv[2]
    name = argv[3] if len(argv) > 3 else username
    password = getpass.getpass(f"Password for {username} (min {MIN_PASSWORD_LENGTH} chars): ")
    if password != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    try:
        hashed = hash_password(password)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f'\n[auth.users.{username}]\nrole = "{role}"\nname = "{name}"\npassword_hash = "{hashed}"')


if __name__ == "__main__":
    main(sys.argv)
