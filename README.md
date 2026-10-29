# Variant Classifier

A Flask web application that uses trained machine-learning models to predict the pathogenic probability of genetic variants. Supports single-variant predictions and batch predictions from CSV files.

## Installation

1. Clone or download the repository.
2. Create and activate a Python virtual environment:

   ```bash
   python -m venv .venv
   ```

   **Windows:**
   ```bash
   .venv\Scripts\activate
   ```

   **Linux/macOS:**
   ```bash
   source .venv/bin/activate
   ```

3. Install the project and its dependencies:

   ```bash
   pip install .
   ```

## Models

Place the trained model files in the `models/` directory. If your model filenames differ from those expected by the application, update the corresponding file paths and names in `app.py`.

## Running the App

From the project root directory, run:

```bash
python app.py
```

Open the local URL displayed in the terminal in your web browser.

**Note:** This application is intended for research and educational use. Predictions should not be treated as independently validated clinical classifications.