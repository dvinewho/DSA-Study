@echo off
REM ==============================================================================
REM run_all.bat — Double-Click Execution Script for Windows
REM Study: Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis
REM ==============================================================================

echo ==============================================================================
echo STARTING FULL BLENDED LEARNING STUDY PIPELINE (DATA INTEGRATION ^& ML SUITES)
echo ==============================================================================
echo.

cd /d "%~dp0"

echo [1/7] Running Stage 1: Data Integration...
python code\data-cleaning\01_data_integration.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in Stage 1! & pause & exit /b %ERRORLEVEL% )

echo [2/7] Running Stage 2: Data Cleansing...
python code\data-cleaning\02_data_cleansing.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in Stage 2! & pause & exit /b %ERRORLEVEL% )

echo [3/7] Running Stage 3: Feature Engineering...
python code\data-cleaning\03_data_transformation.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in Stage 3! & pause & exit /b %ERRORLEVEL% )

echo [4/7] Running Stage 4: PCA Reduction and Sampling...
python code\data-cleaning\04_data_reduction.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in Stage 4! & pause & exit /b %ERRORLEVEL% )

echo [5/7] Running Stage 5: Statistical Analysis and Figures...
python code\data-cleaning\05_data_analysis_and_visualization.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in Stage 5! & pause & exit /b %ERRORLEVEL% )

echo [6/7] Running Supervised Machine Learning Suite (6 Models)...
python code\supervised-ml-algo\main.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in Supervised ML! & pause & exit /b %ERRORLEVEL% )

echo [7/7] Running Unsupervised Machine Learning Suite (Personas)...
python code\unsupervised-ml-algo\main.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in Unsupervised ML! & pause & exit /b %ERRORLEVEL% )

echo.
echo ==============================================================================
echo SUCCESS: ALL 7 PIPELINE STAGES AND ML SUITES COMPLETED WITHOUT ERRORS!
echo Generated tables and charts are saved in: %~dp0output\
echo ==============================================================================
pause

