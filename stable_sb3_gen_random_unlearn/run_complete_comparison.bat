@echo off
REM SNN-DT vs DSF-DT Comparison Script for Windows

echo === SNN-DT vs DSF-DT Comparison ===
echo Starting comparison on CartPole-v1

REM Run the comparison
echo Running comparison...
python src/run_comparison.py --env CartPole-v1 --seed 42

REM Check if comparison was successful
if %ERRORLEVEL% EQU 0 (
    echo Comparison completed successfully!
    
    REM Generate plots
    echo Generating plots...
    python src/plot_comparison.py
    
    REM Display results
    echo Displaying results...
    if exist comparison_results.csv (
        echo === Numerical Results ===
        type comparison_results.csv
    )
    
    if exist training_losses.csv (
        echo === Training Losses ===
        powershell -Command "Get-Content training_losses.csv -Head 10"
        echo ...
        powershell -Command "Get-Content training_losses.csv -Tail 5"
    )
    
    echo === Comparison Complete ===
    echo Results saved to:
    echo - comparison_results.csv
    echo - training_losses.csv
    echo - comparison_results.png
) else (
    echo Comparison failed!
    exit /b 1
)