# Variant Classifier

A Flask web application that uses trained machine-learning models to predict the pathogenic probability of genetic variants. Supports single-variant predictions and batch predictions from CSV files. Predict based on a model of choice or use comparison mode to compare predictions from 3 models. 

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

## Batch Predictions

To predict multiple variants at once, upload a CSV file using the batch prediction form in the web interface.

### Expected CSV Format

The CSV should contain the following column headers:

| Header | Required? | Description |
|---|---|---|
| `mc_variant` | Yes | Molecular consequence of the variant. Must match one of the supported values listed below. |
| `Gene_Length` | No | Length of the gene in base pairs. Missing values are imputed using the batch median, or a default value if no usable values are available. |
| `pLI_Score` | Yes | Gene intolerance score between 0 and 1. |
| `RS_Present` | Yes | Whether an rs identifier is present: `1` for yes, `0` for no. |
| `Allele_Frequency` | No | Maximum allele frequency between 0 and 1. Leave blank if unavailable. |
| `Type` | No | Variant type. Missing values default to `Unknown`. |

Column names are matched without regard to capitalisation or surrounding whitespace.

### Supported Values

**`mc_variant`** must be one of the following molecular consequence feature names:

- `MC_3_prime_UTR_variant`
- `MC_5_prime_UTR_variant`
- `MC_frameshift_variant`
- `MC_genic_downstream_transcript_variant`
- `MC_genic_upstream_transcript_variant`
- `MC_inframe_deletion`
- `MC_inframe_indel`
- `MC_inframe_insertion`
- `MC_initiator_codon_variant`
- `MC_intron_variant`
- `MC_missense_variant`
- `MC_no_sequence_alteration`
- `MC_non-coding_transcript_variant`
- `MC_nonsense`
- `MC_splice_acceptor_variant`
- `MC_splice_donor_variant`
- `MC_stop_lost`
- `MC_synonymous_variant`

**`Type`** must be one of:

- `Complex`
- `Deletion`
- `Insertion`
- `SNV`
- `Unknown`

Missing or blank `Type` values default to `Unknown`. Other unsupported values cause the affected row to be marked invalid.

### Example CSV

Save the following as `variants.csv` and upload it through the batch prediction form:

```csv
mc_variant,Gene_Length,pLI_Score,RS_Present,Allele_Frequency,Type
MC_missense_variant,50000,0.95,1,0.0001,SNV
MC_frameshift_variant,120000,0.99,0,,Deletion
MC_synonymous_variant,30000,0.02,1,0.05,SNV
MC_splice_acceptor_variant,80000,0.85,0,0.00001,Unknown
```

Each row represents one variant. The application validates rows individually, predicts pathogenic probabilities for valid rows, and reports validation errors for invalid rows without discarding the entire batch.

The results include the input features, predicted pathogenic probabilities, and a chart summarising the distribution of predictions.

## Running the App

From the project root directory, run:

```bash
python app.py
```

Open the local URL displayed in the terminal in your web browser.

**Note:** This application is intended for research and educational use. Predictions should not be treated as independently validated clinical classifications.