# OSRCT Benchmark Makefile
# ========================
# Reproducibility commands for the OSRCT Benchmark
#
# Usage:
#   make setup        - Install dependencies
#   make preprocess   - Preprocess raw ManyLabs1 data
#   make generate     - Generate confounded datasets
#   make analyze      - Run causal method evaluation
#   make all          - Run complete pipeline
#   make clean        - Remove generated files
#   make test         - Run tests
#   make help         - Show this help

.PHONY: all setup preprocess generate analyze package clean test help

# Configuration
PYTHON := python3
PIP := pip3
DATA_DIR := data
OUTPUT_DIR := output
RAW_DATA := $(DATA_DIR)/CleanedDataset.sav
PREPROCESSED := $(OUTPUT_DIR)/preprocessed/Manylabs1_data.pkl
CONFOUNDED_DIR := $(OUTPUT_DIR)/confounded_datasets
ANALYSIS_DIR := $(OUTPUT_DIR)/analysis_results
N_JOBS := 4
RANDOM_SEED := 42

# Default target
all: preprocess generate analyze
	@echo "Pipeline complete!"

# Help target
help:
	@echo "OSRCT Benchmark Makefile"
	@echo "========================"
	@echo ""
	@echo "Targets:"
	@echo "  setup        Install Python dependencies"
	@echo "  preprocess   Preprocess raw ManyLabs1 data"
	@echo "  generate     Generate confounded datasets"
	@echo "  analyze      Run causal method evaluation"
	@echo "  all          Run complete pipeline"
	@echo "  clean        Remove generated files"
	@echo "  test         Run tests"
	@echo ""
	@echo "Configuration:"
	@echo "  DATA_DIR     = $(DATA_DIR)"
	@echo "  OUTPUT_DIR   = $(OUTPUT_DIR)"
	@echo "  N_JOBS       = $(N_JOBS)"
	@echo "  RANDOM_SEED  = $(RANDOM_SEED)"

# Environment setup
setup:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Verifying installation..."
	$(PYTHON) -c "import numpy, pandas, scipy, sklearn, matplotlib, seaborn, pyreadstat; print('All dependencies installed successfully!')"

setup-conda:
	@echo "Creating conda environment..."
	conda env create -f environment.yml
	@echo "Activate with: conda activate osrct_benchmark"

# Data preprocessing
preprocess: $(PREPROCESSED)

$(PREPROCESSED): $(RAW_DATA)
	@echo "Preprocessing ManyLabs1 data..."
	@mkdir -p $(OUTPUT_DIR)/preprocessed
	$(PYTHON) code/ml1_preprocess.py \
		--input $(RAW_DATA) \
		--output $(OUTPUT_DIR)/preprocessed
	@echo "Preprocessing complete: $(PREPROCESSED)"

# Dataset generation
generate: $(CONFOUNDED_DIR)/generation_summary.csv

$(CONFOUNDED_DIR)/generation_summary.csv: $(PREPROCESSED)
	@echo "Generating confounded datasets..."
	@mkdir -p $(CONFOUNDED_DIR)
	$(PYTHON) code/generate_confounded_datasets.py \
		--data-path $(PREPROCESSED) \
		--output-dir $(CONFOUNDED_DIR) \
		--seed $(RANDOM_SEED)
	@echo "Generation complete: $(CONFOUNDED_DIR)"

# Analysis
analyze: $(ANALYSIS_DIR)/phase5_findings.md

$(ANALYSIS_DIR)/phase5_findings.md: $(CONFOUNDED_DIR)/generation_summary.csv
	@echo "Running causal method evaluation..."
	@mkdir -p $(ANALYSIS_DIR)
	$(PYTHON) code/run_phase5_analysis.py \
		--confounded-dir $(CONFOUNDED_DIR) \
		--output-dir $(ANALYSIS_DIR)
	@echo "Analysis complete: $(ANALYSIS_DIR)"

# Package for release
package: all
	@echo "Packaging benchmark..."
	$(PYTHON) package_benchmark.py \
		--output-dir osrct_benchmark_release
	@echo "Package complete!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf $(OUTPUT_DIR)
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	find . -name "*.pyc" -delete
	@echo "Clean complete!"

clean-all: clean
	@echo "Removing all outputs including package..."
	rm -rf osrct_benchmark_v1.0
	rm -rf osrct_benchmark_release

# Testing
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "Tests complete!"

test-quick:
	@echo "Running quick tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short -x -q
	@echo "Quick tests complete!"

# Validation
validate:
	@echo "Validating benchmark package..."
	$(PYTHON) -c "from package_benchmark import validate_package; validate_package('osrct_benchmark_v1.0')"

# Download raw data (requires OSF access)
download-data:
	@echo "Downloading ManyLabs1 data from OSF..."
	@mkdir -p $(DATA_DIR)
	@echo "Please download CleanedDataset.sav from https://osf.io/wx7ck/"
	@echo "and place it in $(DATA_DIR)/"

# Generate checksums
checksums:
	@echo "Generating MD5 checksums..."
	find $(CONFOUNDED_DIR) -name "*.csv" -exec md5sum {} \; > checksums.txt
	@echo "Checksums saved to checksums.txt"

# Documentation
docs:
	@echo "Building documentation..."
	# Add sphinx or mkdocs commands here if needed
	@echo "Documentation generation not yet configured"

# Show dataset statistics
stats:
	@echo "Dataset Statistics:"
	@echo "==================="
	@find $(CONFOUNDED_DIR) -name "*.csv" 2>/dev/null | wc -l | xargs echo "Total CSV files:"
	@du -sh $(CONFOUNDED_DIR) 2>/dev/null || echo "Confounded directory not found"
	@du -sh $(ANALYSIS_DIR) 2>/dev/null || echo "Analysis directory not found"
