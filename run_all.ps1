# ==============================================================================
# run_all.ps1 — End-to-End Execution Pipeline Runner
# Course: BSIT 3A – IT 119: Data Scalability and Analytics
# Study: Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis
# Authors: Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano
# Institution: Camarines Norte State College (CNSC)
# ==============================================================================

$ErrorActionPreference = "Stop"

function Print-Banner($title) {
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Yellow
    Write-Host ("=" * 80) -ForegroundColor Cyan
}

$startTime = Get-Date

Print-Banner "STEP 1/7: STAGE 1 — MULTI-SOURCE DATA INTEGRATION"
python code/data-cleaning/01_data_integration.py

Print-Banner "STEP 2/7: STAGE 2 — DATA CLEANSING & SCREENING LOSS AUDIT"
python code/data-cleaning/02_data_cleansing.py

Print-Banner "STEP 3/7: STAGE 3 — FEATURE ENGINEERING & COMPOSITE INDICES"
python code/data-cleaning/03_data_transformation.py

Print-Banner "STEP 4/7: STAGE 4 — PCA DIMENSIONALITY REDUCTION & SAMPLING"
python code/data-cleaning/04_data_reduction.py

Print-Banner "STEP 5/7: STAGE 5 — STATISTICAL ANALYSIS & PUBLICATION FIGURES"
python code/data-cleaning/05_data_analysis_and_visualization.py

Print-Banner "STEP 6/7: SUPERVISED MACHINE LEARNING SUITE (6 ALGORITHMS)"
python code/supervised-ml-algo/main.py

Print-Banner "STEP 7/7: UNSUPERVISED MACHINE LEARNING SUITE (K-MEANS & MEAN-SHIFT)"
python code/unsupervised-ml-algo/main.py

$elapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "  ALL PIPELINES & MACHINE LEARNING MODELS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host ("  Total Execution Time: {0:mm} min {0:ss} sec" -f $elapsed) -ForegroundColor Green
Write-Host "  Outputs saved to: ./output/data-cleaning/, ./output/supervised-ml-algo/, ./output/unsupervised-ml-algo/" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Green

