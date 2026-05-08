Report compilation and usage

This folder contains a LaTeX source for the project report.

Files:
- report.tex -- LaTeX source file
- compile_report.bat -- Windows script to build PDF (requires pdflatex in PATH)
- figures/ -- place screenshots and images here using the exact filenames below

Required figure filenames (place actual screenshots from the UI here):
- figures/detection.png
- figures/removal.png
- figures/visual_comparison.png

Compile (PowerShell):

```powershell
cd report
./compile_report.bat
```

If you don't have a LaTeX distribution installed, install MiKTeX (Windows) or TeX Live (Linux/Mac) and ensure `pdflatex` is in your PATH.
