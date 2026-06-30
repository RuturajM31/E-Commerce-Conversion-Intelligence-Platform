# Streamlit Community Cloud Deployment

## Deployment target

The persistent public deployment target is **Streamlit Community Cloud**.
The application is deployed directly from the GitHub repository after the
master closure commit is merged into `main`.

## Deployment coordinates

| Setting | Value |
|---|---|
| Repository owner | `RuturajM31` |
| Repository | `E-Commerce-Conversion-Intelligence-Platform` |
| Branch | `main` |
| Entrypoint file | `app/Executive_Overview.py` |
| Python version | `3.10` |
| Secrets | None currently required |
| App dependency file | `app/requirements.txt` |
| Streamlit configuration | `.streamlit/config.toml` |

## Why the dependency file is inside `app/`

Streamlit Community Cloud searches the entrypoint directory before the
repository root. The app-local `requirements.txt` therefore installs only the
validated dashboard runtime instead of the heavier training, experiment, and
test environments.

## Browser deployment steps

1. Sign in to Streamlit Community Cloud with GitHub.
2. Switch to the workspace that owns the repository.
3. Click **Create app**.
4. Choose the existing-app option.
5. Select the repository shown above.
6. Select branch `main`.
7. Enter `app/Executive_Overview.py` as the file path.
8. Open **Advanced settings** and select Python `3.10`.
9. Leave Secrets empty because this app currently uses no credentials.
10. Click **Deploy**.

## Verification

After the build completes:

- confirm the Executive Overview loads;
- open each page from the sidebar;
- confirm no missing-file or dependency error appears;
- copy the final `streamlit.app` URL into the portfolio README;
- record the URL in the project handoff and deployment evidence.

## Boundary

The repository can be prepared, tested, committed, pushed, and merged by the
master closure command. The final website authorization and **Deploy** click
must be completed by the project owner in the Streamlit interface.
