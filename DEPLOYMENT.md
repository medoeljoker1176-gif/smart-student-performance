> For the cloud-computing course version (Cloud Run + Firestore, load testing, scalability) see CLOUD_DEPLOYMENT.md.

# Deployment Guide (Streamlit Community Cloud)

## 1. Push to GitHub
1. Create a new **public** repo on github.com (e.g. `smart-student-performance`).
2. Upload the project folder contents (the repo root must contain `app.py` and `requirements.txt`).
   Make sure `models/student_performance_model.pkl` and `data/student_performance.csv` are included.

## 2. Deploy
1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** -> *Deploy a public app from GitHub*.
3. Repository: your repo | Branch: `main` | Main file path: `app.py`.
4. **Advanced settings** -> Python version **3.12** (scikit-learn 1.8 needs 3.11+).
5. Click **Deploy**. First build takes 2-5 minutes. You get `https://<name>.streamlit.app`.

## 2b. Turn on login (optional, recommended for a public URL)
1. Locally run `python tools/make_password_hash.py teacher1 teacher "Dr. Name"` (and again for each student account) and copy the printed block.
2. Streamlit Cloud -> your app -> **Settings -> Secrets**: paste the blocks under the `[auth]` header (see `.streamlit/secrets.toml.example`). Never commit real hashes or passwords to GitHub.
3. Save; the app restarts and shows a sign-in form. No `[auth]` section = no login (demo mode).

## 2c. Real dataset (optional)
Follow "Real data" in README.md, commit `data/uci_student_performance.csv`, `models/uci_student_performance_model.pkl` and `reports/uci/`; the sidebar then offers both datasets.

## 3. If the model gives a version warning
Run `python train_model.py` locally, commit the new `models/student_performance_model.pkl`, push. The app redeploys automatically.

## 4. Alternative: Docker (Render / Hugging Face Spaces)
`Dockerfile` is included. Render: New -> Web Service -> connect repo -> Runtime *Docker*.
Hugging Face: New Space -> SDK *Docker*, set `app_port: 8501` in the Space README header.

## 5. Screenshots to take for the presentation
1. GitHub repo page with the files
2. Streamlit "Create app" form (filled)
3. Advanced settings (Python 3.12)
4. Build logs while deploying
5. App live with the public URL visible in the address bar
6. Prediction page (at-risk student) | 7. AI Explanation | 8. Recommendations | 9. Model Performance

Note: the History page writes a CSV on the server; it resets when the cloud app restarts. This is expected.
