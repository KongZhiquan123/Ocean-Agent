const DESCRIPTION = `
🔴 MANDATORY TOOL for ocean data preprocessing with CNN convergence validation.

CRITICAL: This is the ONLY approved method for preprocessing ocean data. Never write custom preprocessing scripts.

Features:
1. Batch processing of NC files with quality checks
2. Data cleaning, merging, and standardization
3. CNN convergence validation (verifies data is learnable)
4. Automatic generation of validation reports

Use Cases:
- User provides raw ocean data from JAXA, OSTIA, ERA5, CMEMS
- User mentions "prepare data", "preprocess", "clean data"
- Before ANY model training (DiffSR, forecasting, prediction)
- When data quality or convergence needs validation
`

const PROMPT = `
🔴 CRITICAL INSTRUCTIONS:

You MUST use this tool when:
- User provides raw NetCDF (.nc) files
- User mentions data from satellite/ocean sources (JAXA, OSTIA, etc.)
- User asks to prepare/preprocess/clean ocean data
- Before starting ANY training pipeline

NEVER write custom preprocessing scripts. ALWAYS use this tool.

---

This tool runs a TWO-PHASE validation pipeline:

PHASE 1: Statistical Validation (Always runs)
- Missing value analysis
- Outlier detection
- Data distribution checks
- Temporal/spatial continuity

PHASE 2: CNN Convergence Validation (Default: enabled)
- Trains lightweight CNN on preprocessed data
- Verifies data is learnable (loss converges)
- Provides quality score and convergence metrics
- CRITICAL: If this fails, data is NOT ready for production training

---

Output files (in output_dir):
- preprocessed_{variable}.nc - Cleaned and merged data
- validation_report.md - MUST READ THIS and present to user
- validation_results.json - Machine-readable quality metrics

---

YOUR REQUIRED ACTIONS AFTER TOOL COMPLETES:

1. ✅ Read validation_report.md using FileRead tool
2. ✅ Present key findings to user:
   - Convergence status (did CNN loss decrease?)
   - Quality score (0-100, higher is better)
   - Data statistics (shape, missing values, outliers)
   - Recommendations (is data ready for training?)
3. ✅ If validation failed: Explain issues and suggest fixes
4. ✅ If validation passed: Confirm data is ready, suggest next steps
5. ❌ NEVER skip reading the validation report
6. ❌ NEVER proceed to training without presenting validation results

---

VALIDATION INTERPRETATION GUIDE:

✅ GOOD (Ready for training):
- CNN loss decreased during validation
- Quality score > 70
- No critical data issues
- Convergence confirmed

⚠️ NEEDS ATTENTION:
- Quality score 40-70
- Some outliers or missing data
- Weak convergence
→ Present warnings, ask user if they want to proceed

❌ FAILED (NOT ready):
- CNN loss did not converge
- Quality score < 40
- Critical data issues (>20% missing, severe outliers)
→ Explain issues, do NOT proceed to training

---

If CNN validation is unavailable (PyTorch not installed), tool falls back to statistical validation only. Warn user that convergence was not verified.
`
export { DESCRIPTION, PROMPT }