# FEVER Fact Verification Search

An end-to-end Information Retrieval project for FEVER: evidence retrieval for claims using **TF-IDF**, **BM25**, **Sentence Transformers**, and **Reciprocal Rank Fusion (RRF)**.

The implementation supports downloading FEVER claims and the Wikipedia page dump, sentence-level preparation, sparse/dense indexes, RRF fusion, evaluation with Precision@10/Recall/MAP/nDCG, and a Streamlit live search interface.

## Deliverables & Documentation

- **Technical Report (Deliverable 1)**: See [REPORT.md](REPORT.md) for the complete 20-mark technical report with mathematical formulations, performance comparison table, and precision-recall curve analysis.
- **Live Search Interface (Deliverable 2)**: Run `streamlit run app.py` to launch the interactive multi-model retrieval interface.
- **Evaluation Artifacts**:
  - Performance Comparison Table: `artifacts/performance_comparison.csv` and `artifacts/performance_comparison.json`
  - Precision-Recall Dual Graph: `artifacts/precision_recall.png`

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run automated benchmark and evaluation across all 4 models:
$env:PYTHONPATH="."
python scripts/setup_benchmark.py

# Run unit test suite:
pytest

# Launch interactive search interface:
streamlit run app.py
```

For the complete run, download the official Wikipedia archive (large):

```powershell
python -m fever_ir.cli download --full
python -m fever_ir.cli prepare --full
python -m fever_ir.cli build-index --full
python -m fever_ir.cli evaluate --split dev
```

The source is the official FEVER release at https://fever.ai/dataset/fever.html. The project reports FEVER evidence retrieval separately from the claim label: `NOT ENOUGH INFO` claims have no positive evidence set.

## Structure

```text
app.py                         Streamlit search interface
configs/default.yaml           paths and model settings
fever_ir/data                  download and FEVER parsing
fever_ir/retrieval             TF-IDF, BM25, dense, RRF
fever_ir/evaluation            metrics and plots
scripts/run_pipeline.ps1       demo pipeline
tests/                         unit tests
```
