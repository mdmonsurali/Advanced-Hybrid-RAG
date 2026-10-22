"""One-time Google Drive OAuth consent — run on the HOST, not in Docker.

1. In Google Cloud Console: enable the Drive API, create an OAuth client of
   type "Desktop app", download it as secrets/gdrive/credentials.json.
2. pip install google-auth-oauthlib
3. python rag_service/src/mcp_servers/gdrive_auth.py --secrets secrets/gdrive
   A browser opens; approve read-only Drive access. token.json is written
   next to credentials.json and mounted read-only into rag_service.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--secrets",
        default="secrets/gdrive",
        help="directory holding credentials.json (token.json is written here)",
    )
    args = parser.parse_args()

    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets = Path(args.secrets)
    client_file = secrets / "credentials.json"
    if not client_file.exists():
        raise SystemExit(f"{client_file} not found (download the Desktop OAuth client first)")
    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), SCOPES)
    creds = flow.run_local_server(port=0)
    (secrets / "token.json").write_text(creds.to_json(), encoding="utf-8")
    print(f"wrote {secrets / 'token.json'}")


if __name__ == "__main__":
    main()
