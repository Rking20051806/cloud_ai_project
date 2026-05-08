@echo off
REM Compile the LaTeX report into PDF. Requires pdflatex in PATH (MiKTeX/TeX Live).
cd /d %~dp0
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
if exist report.pdf (
  echo PDF built: report.pdf
) else (
  echo Failed to build PDF. Ensure pdflatex is installed and in PATH.
)
pause
