param(
    [switch]$Benchmark,
    [int]$Sample = 2000
)

$env:PYTHONPATH = "."

if ($Benchmark -or -not (Test-Path "artifacts/sparse_index.pkl")) {
    Write-Host "Running automated benchmark pipeline across all 4 models (TF-IDF, BM25, Dense, Hybrid RRF)..."
    python scripts/setup_benchmark.py
} else {
    python -m fever_ir.cli download --sample $Sample
    python -m fever_ir.cli prepare --sample $Sample
    python -m fever_ir.cli build-index --max-passages 50000
    python -m fever_ir.cli evaluate --split dev --max-queries 500
}

Write-Host "Pipeline complete. Launch with: streamlit run app.py"
