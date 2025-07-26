# scripts/explore_raw_data.py

"""Generate a diagnostics report for the raw data files."""

import importlib
import sys
from pathlib import Path
import zipfile
import io
import ast
from collections import Counter
from contextlib import redirect_stdout

REQUIRED_PACKAGES = ["pandas", "pyarrow"]

for pkg in REQUIRED_PACKAGES:
    if importlib.util.find_spec(pkg) is None:
        sys.exit(
            f"Missing required package '{pkg}'. Install dependencies via 'pip install -r requirements.txt'."
        )

import pandas as pd

# --- CONFIGURATION ---
RAW_DATA_ROOT = Path("data/1_raw_source_data")
OUTPUT_FILENAME = "raw_data_diagnostics_report.txt"
SAMPLE_ROWS = 2
SAMPLE_LINES = 5
FILES_TO_IGNORE = ['.DS_Store', 'state.json', 'dataset_info.json', '.arrow']
WIDE_FORMAT_THRESHOLD = 5 # If a column prefix appears this many times, flag as wide.

def print_header(header):
    print("\n" + "="*80)
    print(f"=== {header.upper()} ===")
    print("="*80)

# --- DIAGNOSTICS & REPORTING ---

def diagnose_and_report(df: pd.DataFrame, file_path: Path, summary_findings: list):
    """Analyzes a DataFrame for common structural problems and records findings."""
    print("\n  --- DIAGNOSTICS ---")
    
    # 1. Wide Format Detection
    try:
        prefixes = [col.split('(')[0] for col in df.columns]
        prefix_counts = Counter(prefixes)
        most_common = prefix_counts.most_common(1)[0]
        if most_common[1] > WIDE_FORMAT_THRESHOLD:
            message = f"File appears to be in a WIDE format. Prefix '{most_common[0]}' is repeated {most_common[1]} times."
            print(f"  [!] Issue Detected: {message}")
            summary_findings.append({
                'file': file_path.name,
                'issue': 'Complex Structure (Wide Format)',
                'details': message,
                'action': 'Reshape data from wide to long format (melt).'
            })
            return # Stop further column analysis if it's wide format
    except Exception: pass # Ignore errors in this diagnostic

    # 2. Stringified Object Detection
    for col in df.select_dtypes(include=['object']).columns:
        try:
            sample = df[col].dropna().head(3)
            if sample.empty: continue
            
            # Check if sample values look like dicts or lists and can be parsed
            is_complex = all(val.strip().startswith(('{', '[')) for val in sample)
            if is_complex:
                ast.literal_eval(sample.iloc[0]) # Test parsing on one
                message = f"Column '{col}' appears to contain stringified Python objects (dicts/lists)."
                print(f"  [!] Issue Detected: {message}")
                summary_findings.append({
                    'file': file_path.name,
                    'issue': 'Complex Structure (Stringified Object)',
                    'details': message,
                    'action': 'Use ast.literal_eval to parse this column during ingestion.'
                })
        except (SyntaxError, ValueError, TypeError): continue # Not a valid literal
        except Exception: pass # Ignore other errors

    print("  [+] No major structural issues detected.")

def print_summary_report(findings: list):
    """Prints a summary of all detected issues at the end of the report."""
    print("\n" + "="*80)
    print("--- SUMMARY OF FINDINGS ---")
    print("="*80)

    if not findings:
        print("\nNo programmatic issues were detected in the data files.")
        
    # Manually add non-detectable issues from user analysis for completeness
    findings.append({'file': 'boltmonkey.json, gragroo_train.parquet', 'issue': 'Critical - Licensing', 'action': 'Verify data provenance and license before use.'})
    findings.append({'file': 'All Non-Psychology Sets', 'issue': 'Medium - Domain Mismatch', 'action': 'Consider a staged curriculum learning approach.'})

    # Group findings by issue type for a clean report
    grouped_findings = {}
    for f in findings:
        if f['issue'] not in grouped_findings:
            grouped_findings[f['issue']] = []
        grouped_findings[f['issue']].append(f)

    for issue_type, items in grouped_findings.items():
        print(f"\n--- ISSUE: {issue_type} ---")
        for item in items:
            print(f"  - File(s): {item['file']}")
            if 'details' in item:
                print(f"    Details: {item['details']}")
            print(f"    Action:  {item['action']}")

# --- FILE EXPLORATION LOGIC ---

def explore_dataframe(df: pd.DataFrame, file_path: Path, summary_findings: list):
    """Prints info and runs diagnostics for a pandas DataFrame."""
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {df.columns.to_list()}")
    diagnose_and_report(df, file_path, summary_findings)
    print("\n  --- SAMPLE DATA ---")
    with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.max_colwidth', 80):
        print(df.head(SAMPLE_ROWS))

def explore_text_file(reader, file_path: Path):
    """Prints sample lines from a plain text file."""
    print(f"  - File Type: Plain Text. First {SAMPLE_LINES} lines:")
    for i, line in enumerate(reader):
        if i >= SAMPLE_LINES: break
        print(f"    {line.strip()}")

def explore_zip_archive(zip_path: Path, summary: list):
    print(f"\nInspecting ZIP archive: {zip_path.resolve()}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = [f for f in z.namelist() if Path(f).name not in FILES_TO_IGNORE and not f.startswith('__MACOSX')]
        if not file_list: return print("  - Archive empty or contains only ignored metadata.")
        
        print(f"  - Contains {len(file_list)} file(s): {file_list[:5] if len(file_list) > 5 else file_list}")
        first_data_file = next((f for f in file_list if f.endswith(('.json', '.csv', '.txt', '.jsonl'))), None)

        if not first_data_file: return print("  - No sampleable data files found in archive.")
        
        print(f"\n  --- Sampling first data file from zip: '{first_data_file}' ---")
        with z.open(first_data_file) as internal_file:
            explore_single_file(internal_file, Path(first_data_file), summary)

def explore_single_file(file_handle, file_path: Path, summary: list):
    """Processes a single file object, either from disk or from a zip."""
    try:
        # Handle plain text files first
        if file_path.suffix == '.txt':
            reader = io.TextIOWrapper(file_handle, 'utf-8', errors='ignore')
            explore_text_file(reader, file_path)
            return

        # Handle tabular data with smart encoding
        encoding_used = 'utf-8'
        try:
            content = file_handle.read()
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            encoding_used = 'latin-1'
            text_content = content.decode('latin-1')
        except AttributeError: # Already a text stream
            text_content = file_handle.read()
        
        print(f"  - Text Encoding: {encoding_used}")
        # Use StringIO to treat the decoded text as a file for pandas
        text_stream = io.StringIO(text_content)
        
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_handle) # Parquet needs original handle
        elif file_path.suffix == '.csv':
            df = pd.read_csv(text_stream, on_bad_lines='skip')
        else: # JSON / JSONL
            df = pd.read_json(text_stream, lines=file_path.suffix == '.jsonl')
        
        explore_dataframe(df, file_path, summary)

    except Exception as e:
        print(f"  [ERROR] Could not read or process file '{file_path.name}'. Reason: {e}")

def main():
    """Main function to run the exploration and save the report."""
    script_dir = Path(__file__).parent
    output_file_path = script_dir / OUTPUT_FILENAME
    print(f"Starting data diagnostics. Report will be saved to:\n{output_file_path.resolve()}")
    
    summary_findings = []
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            print("PSYCHOLOGY TUTOR ENGINE - RAW DATA DIAGNOSTICS REPORT")
            for data_dir in sorted(RAW_DATA_ROOT.iterdir()):
                if data_dir.is_dir():
                    print_header(data_dir.name)
                    for file_path in sorted(data_dir.glob("**/*")):
                        if file_path.is_file() and file_path.name not in FILES_TO_IGNORE:
                            if file_path.suffix == '.zip':
                                explore_zip_archive(file_path, summary_findings)
                            else:
                                print(f"\nAnalyzing file: {file_path.resolve()}")
                                with open(file_path, 'rb') as disk_file:
                                    explore_single_file(disk_file, file_path, summary_findings)
            
            print_summary_report(summary_findings)

    print("\nDiagnostics complete. Report successfully saved.")

if __name__ == "__main__":
    main()