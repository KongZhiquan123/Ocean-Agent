const DESCRIPTION = `
Run a complete ocean data preprocessing pipeline with CNN convergence validation.

Features:
1. Batch processing of NC files
2. Data cleaning and merging
3. CNN validation of data convergence
4. Automatic generation of validation reports

Use Cases:
- Preprocessing ocean data from JAXA/OSTIA, etc.
- Validating data quality and convergence
- Preparing training data for super-resolution or forecasting models
`

const PROMPT = `
You are using the OceanPreprocessPipelineTool to run a complete data preprocessing pipeline with CNN validation.

This tool will:
1. Load and process multiple NC files from input_dir
2. Merge them into a single processed file
3. Validate data quality using a lightweight CNN
4. Generate a detailed validation report

Output files (in output_dir):
- preprocessed_{variable}.nc - Processed data file
- validation_report.md - Detailed validation report
- validation_results.json - Machine-readable results

The tool will show you:
- Processing progress
- Data statistics
- Convergence metrics
- Quality scores

If CNN validation is unavailable (PyTorch not installed), it will fall back to basic statistical validation.
`
export { DESCRIPTION, PROMPT }