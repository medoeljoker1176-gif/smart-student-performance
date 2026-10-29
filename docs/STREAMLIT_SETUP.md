# Streamlit setup on Windows (VS Code)

This guide matches the real project layout (`app.py`, `views/`, `src/`, `models/`, `data/`).

## 1. Requirements
* Windows 10/11, **Python 3.11 or newer** (3.12 was used for the non-UI checks). Check with `py --version`.
* VS Code with the *Python* extension (optional but recommended).

## 2. One-time setup (PowerShell, in the project folder)
```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt and `venv\Scripts\activate.bat`.

In VS Code: *Ctrl+Shift+P → Python: Select Interpreter → .\venv\Scripts\python.exe*.

## 3. Start the app
```powershell
python -m streamlit run app.py
```
Streamlit prints a local URL (normally http://localhost:8501) and opens your browser. Stop it with `Ctrl+C`.

The theme and the 5 MB upload limit come from `.streamlit/config.toml`, which Streamlit reads when it is started from the project folder.

## 4. The model file
`models/student_performance_model.pkl` is included and was saved with **scikit-learn 1.8.0**. If the sidebar shows
*"Model not available"* or a version warning, rebuild it on your machine (about 30 seconds, not run automatically):
```powershell
python train_model.py
```
Only load `.pkl` files you created yourself; they are Python pickles.

## 5. Run the tests
```powershell
python -m pytest -q
python -m compileall -q app.py src views tests
```
`tests/test_ui_apptest.py` runs the app headlessly with Streamlit's `AppTest` and is skipped when Streamlit is not installed.

## 6. Troubleshooting
| Symptom | Likely cause and fix |
|---|---|
| `No module named streamlit` | The virtual environment is not active or `pip install -r requirements.txt` did not finish. |
| `streamlit` is not recognised | Use `python -m streamlit run app.py` instead. |
| `ERROR: No matching distribution for scikit-learn>=1.8` | Python is older than 3.11. Install a newer Python and recreate `venv`. |
| Banner about a scikit-learn version mismatch | Run `python train_model.py`. |
| Port 8501 is busy | `python -m streamlit run app.py --server.port 8502` |
| CSV upload rejected | See the *Expected CSV format* box on the Batch page or the README. |
| `History could not be saved` | `data/prediction_history.csv` is open in Excel; close it and predict again. |
