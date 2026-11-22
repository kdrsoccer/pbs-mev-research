#!/usr/bin/env bash
# Copy the regenerated figures the paper uses into 4_paper/figures and compile the paper. Needs pdflatex.
set -e
cd "$(dirname "$0")"
for f in $(grep -o 'figures/fig_[A-Za-z0-9_]*\.\(pdf\|png\)' 4_paper/pbs_paper.tex | sort -u); do [ -e "$f" ] && cp "$f" 4_paper/figures/; done
cd 4_paper
for i in 1 2 3; do pdflatex -interaction=nonstopmode pbs_paper.tex > /dev/null; done; grep -E "Output written" pbs_paper.log
