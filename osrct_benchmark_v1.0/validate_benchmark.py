#!/usr/bin/env python3
"""
Phase 6.6: Final Validation and QA

Comprehensive validation script for the OSRCT Benchmark release.

Usage:
    python validate_benchmark.py
    python validate_benchmark.py --verbose
    python validate_benchmark.py --fix  # Attempt to fix issues
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Try to import pandas, handle gracefully if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available, some checks will be skipped")


class BenchmarkValidator:
    """Comprehensive validator for OSRCT Benchmark package."""

    def __init__(self, benchmark_dir: Path, verbose: bool = False):
        self.benchmark_dir = Path(benchmark_dir)
        self.verbose = verbose
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_dir': str(benchmark_dir),
            'checks': {},
            'warnings': [],
            'errors': [],
            'summary': {}
        }

    def log(self, message: str, level: str = 'info'):
        """Log message with optional verbosity control."""
        if self.verbose or level in ['error', 'warning']:
            prefix = {'info': '  ', 'warning': '⚠️ ', 'error': '❌ ', 'success': '✅ '}
            print(f"{prefix.get(level, '  ')}{message}")

    def check_directory_structure(self) -> bool:
        """Validate required directory structure exists."""
        self.log("Checking directory structure...")

        required_dirs = [
            'confounded_datasets/by_study',
            'ground_truth',
            'metadata',
            'code',
            'analysis_results',
        ]

        all_exist = True
        for dir_path in required_dirs:
            full_path = self.benchmark_dir / dir_path
            exists = full_path.exists() and full_path.is_dir()
            self.results['checks'][f'dir_{dir_path}'] = exists
            if exists:
                self.log(f"  {dir_path}: OK", 'success')
            else:
                self.log(f"  {dir_path}: MISSING", 'error')
                self.results['errors'].append(f"Missing directory: {dir_path}")
                all_exist = False

        return all_exist

    def check_required_files(self) -> bool:
        """Validate required files exist."""
        self.log("Checking required files...")

        required_files = [
            ('README.md', 'Documentation'),
            ('LICENSE', 'License file'),
            ('CITATION.cff', 'Citation metadata'),
            ('requirements.txt', 'Python dependencies'),
            ('environment.yml', 'Conda environment'),
            ('Makefile', 'Build automation'),
            ('ground_truth/rct_ates.csv', 'Ground truth ATEs'),
            ('metadata/dataset_catalog.csv', 'Dataset catalog'),
            ('metadata/data_dictionary.json', 'Data dictionary'),
        ]

        all_exist = True
        for file_path, description in required_files:
            full_path = self.benchmark_dir / file_path
            exists = full_path.exists() and full_path.is_file()
            self.results['checks'][f'file_{file_path}'] = exists
            if exists:
                self.log(f"  {file_path}: OK", 'success')
            else:
                self.log(f"  {file_path}: MISSING ({description})", 'error')
                self.results['errors'].append(f"Missing file: {file_path}")
                all_exist = False

        return all_exist

    def check_datasets(self) -> Tuple[bool, int]:
        """Validate confounded datasets."""
        self.log("Checking datasets...")

        datasets_dir = self.benchmark_dir / 'confounded_datasets' / 'by_study'
        if not datasets_dir.exists():
            self.log("Datasets directory not found", 'error')
            return False, 0

        # Count datasets
        csv_files = list(datasets_dir.rglob("*.csv"))
        n_datasets = len(csv_files)

        self.results['checks']['dataset_count'] = n_datasets
        self.log(f"  Found {n_datasets} datasets")

        # Check expected count (15 studies × 5 patterns × 7 betas = 525)
        expected_min = 500
        if n_datasets >= expected_min:
            self.log(f"  Dataset count OK (>= {expected_min})", 'success')
            self.results['checks']['dataset_count_valid'] = True
        else:
            self.log(f"  Expected >= {expected_min} datasets", 'warning')
            self.results['warnings'].append(f"Only {n_datasets} datasets found")
            self.results['checks']['dataset_count_valid'] = False

        # Check study directories
        expected_studies = [
            'anchoring1', 'anchoring2', 'anchoring3', 'anchoring4',
            'allowedforbidden', 'contact', 'flag', 'gainloss', 'gambfal',
            'iat', 'money', 'quote', 'reciprocity', 'scales', 'sunk'
        ]

        found_studies = [d.name for d in datasets_dir.iterdir() if d.is_dir()]
        missing_studies = set(expected_studies) - set(found_studies)

        if missing_studies:
            self.log(f"  Missing studies: {missing_studies}", 'warning')
            self.results['warnings'].append(f"Missing studies: {missing_studies}")

        return n_datasets >= expected_min, n_datasets

    def check_ground_truth(self) -> bool:
        """Validate ground truth file."""
        self.log("Checking ground truth...")

        gt_path = self.benchmark_dir / 'ground_truth' / 'rct_ates.csv'
        if not gt_path.exists():
            self.log("Ground truth file not found", 'error')
            return False

        if not PANDAS_AVAILABLE:
            self.log("Pandas not available, skipping detailed check", 'warning')
            return True

        try:
            gt = pd.read_csv(gt_path)

            # Check required columns
            required_cols = ['study', 'ate', 'ate_se', 'n_total']
            missing_cols = [c for c in required_cols if c not in gt.columns]

            if missing_cols:
                self.log(f"  Missing columns: {missing_cols}", 'error')
                self.results['errors'].append(f"Ground truth missing columns: {missing_cols}")
                return False

            # Check study count
            n_studies = len(gt)
            self.log(f"  {n_studies} studies with ground truth", 'success')
            self.results['checks']['ground_truth_studies'] = n_studies

            # Check for missing values
            if gt[required_cols].isnull().any().any():
                self.log("  Contains missing values", 'warning')
                self.results['warnings'].append("Ground truth has missing values")

            return True

        except Exception as e:
            self.log(f"  Error reading ground truth: {e}", 'error')
            self.results['errors'].append(f"Ground truth read error: {e}")
            return False

    def check_metadata(self) -> bool:
        """Validate metadata files."""
        self.log("Checking metadata...")

        # Check catalog
        catalog_path = self.benchmark_dir / 'metadata' / 'dataset_catalog.csv'
        if catalog_path.exists() and PANDAS_AVAILABLE:
            try:
                catalog = pd.read_csv(catalog_path)
                self.log(f"  Catalog: {len(catalog)} entries", 'success')
                self.results['checks']['catalog_entries'] = len(catalog)
            except Exception as e:
                self.log(f"  Catalog error: {e}", 'error')
                return False

        # Check data dictionary
        dd_path = self.benchmark_dir / 'metadata' / 'data_dictionary.json'
        if dd_path.exists():
            try:
                with open(dd_path) as f:
                    dd = json.load(f)
                self.log(f"  Data dictionary: {len(dd)} sections", 'success')
                self.results['checks']['data_dict_sections'] = len(dd)
            except Exception as e:
                self.log(f"  Data dictionary error: {e}", 'error')
                return False

        # Check checksums
        checksum_path = self.benchmark_dir / 'metadata' / 'checksums.txt'
        if checksum_path.exists():
            with open(checksum_path) as f:
                n_checksums = len(f.readlines())
            self.log(f"  Checksums: {n_checksums} files", 'success')
            self.results['checks']['checksum_count'] = n_checksums
        else:
            self.log("  Checksums file not found", 'warning')

        return True

    def check_code(self) -> bool:
        """Validate code files."""
        self.log("Checking code...")

        code_dir = self.benchmark_dir / 'code'
        if not code_dir.exists():
            self.log("Code directory not found", 'error')
            return False

        py_files = list(code_dir.glob("*.py"))
        self.log(f"  Found {len(py_files)} Python files", 'success')
        self.results['checks']['code_files'] = len(py_files)

        # Check for syntax errors (basic)
        for py_file in py_files:
            try:
                with open(py_file) as f:
                    compile(f.read(), py_file, 'exec')
                self.log(f"    {py_file.name}: syntax OK")
            except SyntaxError as e:
                self.log(f"    {py_file.name}: syntax error - {e}", 'error')
                self.results['errors'].append(f"Syntax error in {py_file.name}")
                return False

        return True

    def check_analysis_results(self) -> bool:
        """Validate analysis results."""
        self.log("Checking analysis results...")

        analysis_dir = self.benchmark_dir / 'analysis_results'
        if not analysis_dir.exists():
            self.log("Analysis results not found", 'warning')
            return True  # Not critical

        # Check figures
        figures_dir = analysis_dir / 'figures'
        if figures_dir.exists():
            figures = list(figures_dir.glob("*.png")) + list(figures_dir.glob("*.pdf"))
            self.log(f"  Figures: {len(figures)}", 'success')
            self.results['checks']['figure_count'] = len(figures)

        # Check method evaluation
        method_dir = analysis_dir / 'method_evaluation'
        if method_dir.exists():
            csv_files = list(method_dir.glob("*.csv"))
            self.log(f"  Method evaluation files: {len(csv_files)}", 'success')
            self.results['checks']['method_eval_files'] = len(csv_files)

        return True

    def verify_sample_checksums(self, n_samples: int = 10) -> bool:
        """Verify MD5 checksums for a sample of datasets."""
        self.log(f"Verifying checksums (sample of {n_samples})...")

        checksum_path = self.benchmark_dir / 'metadata' / 'checksums.txt'
        if not checksum_path.exists():
            self.log("Checksums file not found", 'warning')
            return True

        try:
            with open(checksum_path) as f:
                lines = f.readlines()

            # Sample random lines
            import random
            sample_lines = random.sample(lines, min(n_samples, len(lines)))

            all_valid = True
            for line in sample_lines:
                parts = line.strip().split('  ')
                if len(parts) != 2:
                    continue

                expected_md5, rel_path = parts
                full_path = self.benchmark_dir / 'confounded_datasets' / 'by_study' / rel_path

                if not full_path.exists():
                    self.log(f"  {rel_path}: FILE NOT FOUND", 'error')
                    all_valid = False
                    continue

                # Compute actual MD5
                hash_md5 = hashlib.md5()
                with open(full_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                actual_md5 = hash_md5.hexdigest()

                if actual_md5 == expected_md5:
                    self.log(f"  {rel_path}: OK")
                else:
                    self.log(f"  {rel_path}: MISMATCH", 'error')
                    all_valid = False

            if all_valid:
                self.log(f"  All {len(sample_lines)} samples verified", 'success')

            return all_valid

        except Exception as e:
            self.log(f"  Checksum verification error: {e}", 'error')
            return False

    def calculate_total_size(self) -> int:
        """Calculate total size of benchmark package."""
        total_size = 0
        for f in self.benchmark_dir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
        return total_size

    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("=" * 60)
        print("OSRCT Benchmark Validation")
        print("=" * 60)
        print(f"\nBenchmark directory: {self.benchmark_dir}\n")

        checks_passed = []

        # Run each check
        checks_passed.append(('Directory structure', self.check_directory_structure()))
        checks_passed.append(('Required files', self.check_required_files()))

        ds_valid, n_datasets = self.check_datasets()
        checks_passed.append(('Datasets', ds_valid))

        checks_passed.append(('Ground truth', self.check_ground_truth()))
        checks_passed.append(('Metadata', self.check_metadata()))
        checks_passed.append(('Code', self.check_code()))
        checks_passed.append(('Analysis results', self.check_analysis_results()))
        checks_passed.append(('Checksum verification', self.verify_sample_checksums()))

        # Calculate summary
        total_size = self.calculate_total_size()
        size_mb = total_size / (1024 * 1024)

        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        all_passed = True
        for check_name, passed in checks_passed:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check_name}: {status}")
            if not passed:
                all_passed = False

        print(f"\n  Datasets: {n_datasets}")
        print(f"  Total size: {size_mb:.1f} MB")
        print(f"  Errors: {len(self.results['errors'])}")
        print(f"  Warnings: {len(self.results['warnings'])}")

        if self.results['errors']:
            print("\n  Errors:")
            for err in self.results['errors'][:5]:
                print(f"    - {err}")

        if self.results['warnings']:
            print("\n  Warnings:")
            for warn in self.results['warnings'][:5]:
                print(f"    - {warn}")

        # Final result
        print("\n" + "=" * 60)
        if all_passed:
            print("VALIDATION PASSED ✅")
        else:
            print("VALIDATION FAILED ❌")
        print("=" * 60)

        # Save results
        self.results['summary'] = {
            'all_passed': all_passed,
            'total_datasets': n_datasets,
            'total_size_mb': size_mb,
            'n_errors': len(self.results['errors']),
            'n_warnings': len(self.results['warnings'])
        }

        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Validate OSRCT Benchmark package'
    )
    parser.add_argument(
        '--benchmark-dir', '-d',
        type=str,
        default='.',
        help='Path to benchmark directory'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save validation results to JSON file'
    )

    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.exists():
        print(f"Error: Directory not found: {benchmark_dir}")
        return 1

    validator = BenchmarkValidator(benchmark_dir, verbose=args.verbose)
    passed = validator.run_all_checks()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(validator.results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
