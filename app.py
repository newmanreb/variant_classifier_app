"""
Flask web application for a machine-learning variant classifier. 

This application provides a web interface for submitting individual
genetic variant features or uploading a CSV file containing multiple
variants. It uses trained random forest models to estimate the 
probability that each variant belongs to the pathogenic class. 

The application validates input data, constructs the feature
representation expected by the training models, generates predictions, 
and displays the results through the web interface. 
"""

import os
import pickle as pk
import joblib
import pandas as pd
import numpy as np
import json
from flask import Flask, render_template, request

# Create a Flask application object and assign it to a variable 'app'
app = Flask(__name__)

# Enable automatic reloading of templates when files change 
# so interface edits can be tested without manual restart 
app.config['TEMPLATES_AUTO_RELOAD'] = True

#############################################################################
### DEFINE MODEL AND FEATURES 
#############################################################################

# Load both pipeline models securely into a registry
# I have selected the original random forest model and the optimised random forest model 
MODELS = {
    'rf_model': pk.load(open('models/rf_model.sav', 'rb')),
    'optimised_random_forest': joblib.load('models/optimised_random_forest.joblib')
}

# Categorical features in exact training order
mc_variants = [
    'MC_3_prime_UTR_variant',
    'MC_5_prime_UTR_variant',
    'MC_frameshift_variant',
    'MC_genic_downstream_transcript_variant',
    'MC_genic_upstream_transcript_variant',
    'MC_inframe_deletion',
    'MC_inframe_indel',
    'MC_inframe_insertion',
    'MC_initiator_codon_variant',
    'MC_intron_variant',
    'MC_missense_variant',
    'MC_no_sequence_alteration',
    'MC_non-coding_transcript_variant',
    'MC_nonsense',
    'MC_splice_acceptor_variant',
    'MC_splice_donor_variant',
    'MC_stop_lost',
    'MC_synonymous_variant'
]

numeric_features = [
    'Gene_Length',
    'pLI_Score',
    'RS_Present',
    'AF_MAX',
    'AF_Missing',
    'Type_Complex',
    'Type_Deletion',
    'Type_Insertion',
    'Type_SNV',
    'Type_Unknown'
]
DEFAULT_GENE_LENGTH = 1000

variant_types = [
    'Type_Complex',
    'Type_Deletion',
    'Type_Insertion',
    'Type_SNV',
    'Type_Unknown'
]

all_feature_names = mc_variants + numeric_features

# Check that the feature list contains exactly 28 entries and that
# every feature name is unique. This helps catch accidental omissions
# or duplicate names during changes to the feature definitions. 
assert len(all_feature_names) == 28
assert len(set(all_feature_names)) == 28

#############################################################################
### HELPER FUNCTIONS 
#############################################################################

def parse_allele_frequency(value):
    """
    Converts an allele frequency value into AF_MAX and AF_Missing
    features expected by the models. 

    Parameters
    ----------
    value : float, str, or None
        The allele frequency to process. The value may be a numeric
        value, a string representing a number, or a missing value.
        An allele frequency of zero is treated as a known frequency,
        not as missing data.

    Returns
    -------
    tuple[float, int]
        A tuple containing:
        - AF_MAX (float): The supplied allele frequency, or 0.0 if
          the frequency is missing.
        - AF_Missing (int): 1 if the frequency is missing, or 0 if
          the frequency is known.

    Raises
    ------
    ValueError
        If the supplied allele frequency cannot be converted to a
        float, or if its value falls outside the range 0 to 1.

    Notes
    -----
    Missing values are represented using AF_MAX = 0.0 and
    AF_Missing = 1. Known frequencies, including zero, are represented
    using AF_MAX equal to the supplied value and AF_Missing = 0.

    This distinction allows the models to distinguish between a
    variant with a known frequency of zero and a variant for which
    frequency information is unavailable.
    """
    if value is None or pd.isna(value) or str(value).strip() == '':
        return 0.0, 1

    af_max = float(value)

    if not 0 <= af_max <= 1:
        raise ValueError("Allele frequency must be between 0 and 1.")

    return af_max, 0

def normalise_variant_type(value):
    """
    Validate and normalise a variant type into its corresponding
    Type_* feature name for use by the models.

    Parameters
    ----------
    value : str or None
        The variant type to process. The value may be supplied as a
        plain variant type (e.g. 'SNV') or as a complete feature name
        (e.g. 'Type_SNV'). Missing or blank values are treated as
        'Unknown'.

    Returns
    -------
    str
        The normalised variant type feature name, prefixed with
        'Type_'. The result will be one of:
        'Type_Complex', 'Type_Deletion', 'Type_Insertion',
        'Type_SNV', or 'Type_Unknown'.

    Raises
    ------
    ValueError
        If the supplied value does not match one of the five
        supported variant types after whitespace removal and
        removal of an optional 'Type_' prefix.

    Notes
    -----
    This function accepts both the plain variant type labels used
    in CSV input and the corresponding feature names used internally
    by the application.

    Missing or blank values default to 'Type_Unknown'. This ensures
    that a missing variant type is represented explicitly rather
    than being treated as a known variant category.

    The returned feature name is intended to match the naming
    convention used by the application's variant-type features.
    """
    if value is None or pd.isna(value) or str(value).strip() == '':
        return 'Type_Unknown'

    value = str(value).strip()

    # Accept either the CSV values or the full feature names.
    if value.startswith('Type_'):
        value = value[5:]

    allowed_types = {
        'Complex',
        'Deletion',
        'Insertion',
        'SNV',
        'Unknown'
    }

    if value not in allowed_types:
        raise ValueError(
            f"Invalid variant type '{value}'. "
            f"Use one of: {', '.join(sorted(allowed_types))}."
        )

    return f'Type_{value}'

def make_feature_row(mc_variant, gene_length, pli_score,
                     rs_present, allele_frequency, variant_type):
    """
    Validate variant information and construct a single-row DataFrame
    containing the 28 input features required by the trained models.

    Parameters
    ----------
    mc_variant : str
        The molecular consequence of the variant. Must match one
        of the supported values in mc_variants.

    gene_length : float or int
        Length of the gene associated with the variant. Must be
        greater than 0 and no greater than 2,500,000.

    pli_score : float or int
        The pLI score associated with the gene, representing a
        measure of intolerance to loss-of-function variation.
        Must be between 0 and 1, inclusive.

    rs_present : float or int
        Indicator of whether an rs identifier is present.
        Must be 1 if present or 0 if absent.

    allele_frequency : float, str, or None
        The maximum allele frequency to process. Passed to
        parse_allele_frequency(), which converts it into AF_MAX
        and AF_Missing features.

    variant_type : str or None
        The variant type, such as 'SNV', 'Deletion', 'Insertion',
        'Complex', or 'Unknown'. Passed to normalise_variant_type()
        for validation and standardisation.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row and 28 columns, containing the
        validated and encoded feature values. Columns follow the
        order specified by all_feature_names.

    Raises
    ------
    ValueError
        If the molecular consequence is not supported, if gene
        length, pLI score, or rs_present falls outside its allowed
        range, or if allele frequency or variant type is invalid.

    Notes
    -----
    Molecular consequence and variant type are represented using
    one-hot encoding. Each category is represented by a group of
    binary features, with the selected category assigned 1 and
    the other categories assigned 0.

    The remaining features are numerical values, including gene
    length, pLI score, rs identifier presence, maximum allele
    frequency, and the allele-frequency missingness indicator.

    The feature values are assembled in the order defined by
    all_feature_names to maintain compatibility with the input
    structure expected by the trained models.

    This function is shared by single-variant and batch prediction
    workflows to keep feature construction consistent.
    """
    if mc_variant not in mc_variants:
        raise ValueError(f"Invalid MC variant: {mc_variant}")

    gene_length = float(gene_length)
    pli_score = float(pli_score)
    rs_present = float(rs_present)

    if not 0 < gene_length <= 2500000:
        raise ValueError("Gene Length must be greater than 0 and at most 2,500,000.")

    if not 0 <= pli_score <= 1:
        raise ValueError("pLI Score must be between 0 and 1.")

    if rs_present not in (0, 1):
        raise ValueError("RS Present must be 0 or 1.")

    af_max, af_missing = parse_allele_frequency(allele_frequency)
    chosen_type = normalise_variant_type(variant_type)

    # One-hot encoding for mc_values and type_values
    mc_values = [
        1 if variant == mc_variant else 0
        for variant in mc_variants
    ]

    type_values = [
        1 if feature == chosen_type else 0
        for feature in variant_types
    ]

    values = (
        mc_values
        + [
            gene_length,
            pli_score,
            rs_present,
            af_max,
            af_missing
        ]
        + type_values
    )

    return pd.DataFrame([values], columns=all_feature_names)

def format_features(form_data):
    """
    Extract input values from the single-prediction web form and
    pass them to make_feature_row() to validate and construct
    the model's input features.

    Parameters
    ----------
    form_data : flask.Request.form or similar mapping
        A dictionary-like object containing the values submitted
        through the single-prediction HTML form. Expected fields
        include 'mc_variant', 'gene_length', 'pli_score',
        'rs_present', 'allele_frequency', and 'variant_type'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing one row and 28 columns, with the
        validated and encoded features required by the trained
        machine-learning models. The columns follow the order
        specified by all_feature_names.

    Raises
    ------
    ValueError
        If the submitted values fail the validation performed by
        make_feature_row(), parse_allele_frequency(), or
        normalise_variant_type().

    Notes
    -----
    The get() method retrieves each form value by its field name.
    Default values are supplied when certain fields are absent:
    empty strings for most fields and 'Unknown' for variant_type.

    The function delegates input validation, numeric conversion,
    categorical encoding, and DataFrame construction to
    make_feature_row(), avoiding duplication of that logic.

    This helper is intended for the single-prediction workflow.
    Batch CSV inputs are processed separately.
    """
    return make_feature_row(
        mc_variant=form_data.get('mc_variant', '').strip(),
        gene_length=form_data.get('gene_length', ''),
        pli_score=form_data.get('pli_score', ''),
        rs_present=form_data.get('rs_present', ''),
        allele_frequency=form_data.get('allele_frequency', ''),
        variant_type=form_data.get('variant_type', 'Unknown')
    )

def get_pathogenic_probability(model, features):
    """
    Return the model's predicted probability for the pathogenic class.

    Parameters
    ----------
    model : fitted classifier or pipeline
        A trained model that supports predict_proba() and exposes
        its class labels through classes_.
    features : pandas.DataFrame
        A single row containing the model's expected input features.

    Returns
    -------
    float
        The predicted probability associated with class 1
        (pathogenic), between 0 and 1.

    Notes
    -----
    The index of class 1 is looked up explicitly rather than
    assuming that it is always the second probability returned.
    """
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("The selected model does not contain pathogenic class 1.")

    pathogenic_index = classes.index(1)
    return float(probabilities[pathogenic_index] * 100)

#############################################################################
### FLASK APP 
#############################################################################

# app.context_processor is a Flask decorator that makes variables available
# to every template rendered by the application 
@app.context_processor
def inject_global_vars():
    """
    Make shared variables available to all Flask templates.

    This function acts as a Flask context processor, providing
    molecular consequence (MC) options, variant type options,
    numeric feature names, and available model names to templates
    without requiring each route to pass them individually.

    Returns
    -------
    dict
        A dictionary containing the shared variables that will
        be available in the context of every Flask template.
    """
    return dict(
        mc_variants=mc_variants,
        numeric_features=numeric_features,
        variant_types=variant_types,
        available_models=list(MODELS.keys())
    )

# app.route() is a Flask route decorator that tells Flask which URL
# should trigger the function immediately below it. 
@app.route('/')
def home():
    """
    Defines what happens when someone visits the homepage of the
    Flask app. Uses Flask function render_template to find the template
    in the templates/ directory and processes any Jinja template it 
    contains, returning the rendered HTML. 
    """
    return render_template('index.html')

# POST requests are a type of HTTP request that allows the user to 
# submit data to the server. 
@app.route('/predict', methods=['POST'])
def predict():
    """
    Process a single-variant prediction submitted through the web form.

    This route accepts POST requests containing the variant's input
    features and the selected machine-learning model. It validates
    the selected model, formats and validates the submitted features,
    and calculates the probability that the variant is pathogenic.

    The prediction result or any error message is passed to the
    index.html template for display. The selected model is also
    passed to the template so that the user's model choice is retained.

    Returns
    -------
    flask.Response
        A rendered HTML response containing the prediction result
        or an error message, along with the selected model.

    Notes
    -----
    If no model is specified, the route defaults to 'rf_model'.
    Invalid model selections and errors encountered during feature
    processing or prediction are caught and displayed to the user.

    This route handles single-variant predictions. Batch predictions
    are handled by a separate route.
    """
    selected_model_key = request.form.get(
        'model_choice', 'rf_model'
    )

    try:
        chosen_model = MODELS.get(selected_model_key)

        if chosen_model is None:
            raise ValueError("Please select a valid prediction model.")

        final_features = format_features(request.form)

        prob_pathogenic = get_pathogenic_probability(
            chosen_model,
            final_features
        )

        result_text = (
            f"[{selected_model_key}] "
            f"Pathogenic Probability: {prob_pathogenic:.2f}%"
        )

        return render_template(
            'index.html',
            result=result_text,
            selected_model=selected_model_key
        )

    except Exception as e:
        return render_template(
            'index.html',
            result=f"Error: {str(e)}",
            selected_model=selected_model_key
        )

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Process a batch of genetic variants uploaded as a CSV file.

    This route accepts a POST request containing a CSV file and
    the selected machine-learning model. It validates the upload,
    reads the CSV data, identifies the required input columns, and
    prepares the data for prediction.

    Column names are matched without regard to capitalisation or
    surrounding whitespace. Missing gene lengths are imputed using
    the median gene length in the uploaded file, or
    DEFAULT_GENE_LENGTH if no usable gene lengths are available.
    Other required features are checked for missing or invalid
    values, and missing variant types default to 'Unknown'.

    Each row is validated individually. Invalid rows are retained
    in the results with an explanation of the validation errors,
    while valid rows are converted into the feature representation
    expected by the selected model. The model then predicts the
    pathogenic probability for all valid rows together.

    The route also calculates the distribution of predicted
    pathogenic probabilities across five percentage ranges and
    prepares the results and chart data for display in the
    index.html template.

    Returns
    -------
    flask.Response
        A rendered HTML response containing the batch prediction
        results and probability distribution chart data, or an
        error message if the upload or batch processing fails.

    Notes
    -----
    If no model is specified, the route defaults to 'rf_model'.
    The required input columns are mc_variant, pLI_Score, and
    RS_Present. Gene_Length is optional because missing or absent
    gene lengths can be imputed. Allele_Frequency and Type are
    also optional; missing allele frequencies are treated as
    unavailable data, and missing variant types default to
    'Unknown'.

    Rows that fail validation are excluded from prediction but
    remain in the output with an explanation of the problem.
    Valid rows receive a pathogenic probability expressed as a
    percentage.

    This route handles batch predictions. Single-variant
    predictions are handled by a separate route.
    """
    selected_model_key = request.form.get(
        'model_choice', 'rf_model'
    )

    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template(
            'index.html',
            batch_error="No file selected.",
            selected_model=selected_model_key
        )

    try:
        batch_model = MODELS.get(selected_model_key)

        if batch_model is None:
            raise ValueError("Please select a valid prediction model.")

        df = pd.read_csv(request.files['file'])

        if df.empty:
            raise ValueError("The uploaded CSV contains no data rows.")

        # Find columns without being sensitive to capitalisation or spaces.
        def get_col(col_name):
            for column in df.columns:
                if str(column).strip().lower() == col_name.lower():
                    return column
            return None

        c_gene = get_col('Gene_Length')
        c_pli = get_col('pLI_Score')
        c_rs = get_col('RS_Present')
        c_mc = get_col('mc_variant')
        c_af = get_col('Allele_Frequency')
        c_type = get_col('Type')

        # Gene length: preserve the existing batch-median imputation approach.
        if c_gene is not None:
            gene_values = pd.to_numeric(df[c_gene], errors='coerce')
            median_gene = gene_values.median()

            if pd.isna(median_gene):
                median_gene = DEFAULT_GENE_LENGTH

            df['Gene_Length'] = gene_values.fillna(median_gene)
        else:
            df['Gene_Length'] = DEFAULT_GENE_LENGTH

        # Prepare the columns used for output and validation.
        df['pLI_Score'] = (
            pd.to_numeric(df[c_pli], errors='coerce')
            if c_pli is not None
            else np.nan
        )

        df['RS_Present'] = (
            pd.to_numeric(df[c_rs], errors='coerce')
            if c_rs is not None
            else np.nan
        )

        df['mc_variant'] = (
            df[c_mc].fillna('').astype(str).str.strip()
            if c_mc is not None
            else ''
        )

        # Keep the original allele-frequency column for the results table.
        df['Allele_Frequency'] = (
            df[c_af] if c_af is not None else np.nan
        )

        # Missing Type becomes Unknown.
        df['Type'] = (
            df[c_type].fillna('').astype(str).str.strip()
            if c_type is not None
            else 'Unknown'
        )

        results_col = [None] * len(df)
        valid_positions = []
        feature_rows = []

        # Validate each row and build its features.
        for position, (_, row) in enumerate(df.iterrows()):
            problems = []

            mc_value = row['mc_variant']
            gene_value = row['Gene_Length']
            pli_value = row['pLI_Score']
            rs_value = row['RS_Present']
            af_value = row['Allele_Frequency']
            type_value = row['Type']

            if not mc_value or mc_value not in mc_variants:
                problems.append('MC variant missing or invalid')

            if pd.isna(gene_value):
                problems.append('Gene Length missing or invalid')
            elif not 0 < gene_value <= 2500000:
                problems.append('Gene Length must be > 0 and <= 2,500,000')

            if pd.isna(pli_value):
                problems.append('pLI Score missing or invalid')
            elif not 0 <= pli_value <= 1:
                problems.append('pLI Score must be between 0 and 1')

            if pd.isna(rs_value):
                problems.append('RS Present missing or invalid')
            elif rs_value not in (0, 1):
                problems.append('RS Present must be 0 or 1')

            # A blank/NaN AF is allowed. A nonblank, nonnumeric AF is not.
            if af_value is not None and str(af_value).strip() != '':
                if not pd.isna(af_value):
                    try:
                        parsed_af = float(af_value)
                        if not 0 <= parsed_af <= 1:
                            problems.append(
                                'Allele Frequency must be between 0 and 1'
                            )
                    except (TypeError, ValueError):
                        problems.append('Allele Frequency is not a valid number')

            try:
                normalise_variant_type(type_value)
            except ValueError as error:
                problems.append(str(error))

            if problems:
                results_col[position] = (
                    "Invalid row: " + "; ".join(problems)
                )
                continue

            try:
                features = make_feature_row(
                    mc_variant=mc_value,
                    gene_length=gene_value,
                    pli_score=pli_value,
                    rs_present=rs_value,
                    allele_frequency=af_value,
                    variant_type=type_value
                )

                feature_rows.append(features.iloc[0].tolist())
                valid_positions.append(position)

            except (TypeError, ValueError) as error:
                results_col[position] = f"Invalid row: {error}"

        # Predict all valid rows together.
        raw_probabilities = []

        if valid_positions:
            final_features_df = pd.DataFrame(
                feature_rows,
                columns=all_feature_names
            )

            probabilities = batch_model.predict_proba(final_features_df)
            classes = list(batch_model.classes_)

            if 1 not in classes:
                raise ValueError(
                    "The selected model does not contain pathogenic class 1."
                )

            pathogenic_index = classes.index(1)
            probs_pct = probabilities[:, pathogenic_index] * 100

            raw_probabilities = probs_pct.tolist()

            for position, probability in zip(valid_positions, probs_pct):
                results_col[position] = f"{probability:.2f}%"

        # Probability distribution chart.
        bins = [0, 20, 40, 60, 80, 100.01]
        bin_labels = [
            '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'
        ]

        distribution_counts = [0] * len(bin_labels)

        if raw_probabilities:
            categorized = pd.cut(
                raw_probabilities,
                bins=bins,
                labels=bin_labels,
                right=False
            )
            counts = categorized.value_counts().reindex(
                bin_labels, fill_value=0
            )
            distribution_counts = counts.astype(int).tolist()

        chart_data = {
            'labels': bin_labels,
            'counts': distribution_counts,
            'valid_count': len(valid_positions),
            'invalid_count': len(df) - len(valid_positions)
        }

        # Format results for display.
        df['Pathogenic_Probability_Pct'] = results_col
        df['Gene_Length'] = df['Gene_Length'].round(0).astype(int)

        output_cols = [
            'mc_variant',
            'Gene_Length',
            'pLI_Score',
            'RS_Present',
            'Allele_Frequency',
            'Type',
            'Pathogenic_Probability_Pct'
        ]

        formatted_results = df[output_cols].to_dict(orient='records')

        return render_template(
            'index.html',
            batch_results=formatted_results,
            chart_data=json.dumps(chart_data),
            selected_model=selected_model_key
        )

    except Exception as e:
        return render_template(
            'index.html',
            batch_error=f"Error processing CSV: {str(e)}",
            selected_model=selected_model_key
        )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)