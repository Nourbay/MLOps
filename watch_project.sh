#!/bin/bash

echo "==================================================="
echo "         Starting Project Watcher                "
echo "   Changes in .py, .csv, Makefile, requirements.txt"
echo "       will trigger the Local CI (make ci-local)   "
echo "==================================================="
echo "Press Ctrl+C to stop."
echo ""

# Find relevant files, exclude the virtual environment folder
find . -path "./venv" -prune -o \
       -type f \( -name "*.py" -o -name "*.csv" -o -name "Makefile" -o -name "requirements.txt" \) \
       -print | \
    entr -rc make ci-local

