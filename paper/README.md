# Paper draft (ICLR-style)

Working LaTeX draft for **Transcoder-Guided Outlier Selection for Low-Bit LLM Quantization** (T-DSO vs TaCQ).

## Structure

| File | Content |
|------|---------|
| `main.tex` | Title, abstract, document skeleton |
| `sections/introduction.tex` | Problem, motivation, contributions |
| `sections/research_questions.tex` | RQ1--RQ6 with experiment mapping |
| `sections/related_work.tex` | PTQ, TaCQ, interpretability, Spider |
| `sections/method.tex` | T-DSO algorithm |
| `sections/experimental_setup.tex` | Fair comparison protocol |
| `sections/results.tex` | Tables (placeholders) |
| `sections/analysis.tex` | Planned analyses |
| `sections/limitations.tex` | |
| `sections/conclusion.tex` | |
| `sections/appendix.tex` | Hyperparams, ablations |
| `references.bib` | Starter bibliography |

## Build

**On the cluster** (tectonic bundled under `paper/.bin/` after first compile):

```bash
cd paper
.bin/tectonic -X compile main.tex   # produces main.pdf
```

**Locally** with a full TeX install:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires: `pdflatex`, `bibtex`, packages `times`, `natbib`, `booktabs`, `hyperref`, `todonotes`.

Before submission, swap `iclr2025_conference.sty` for the [official ICLR template](https://github.com/ICLR/Master-Template) or NeurIPS style.

## TODO markers

Red `[TODO: ...]` tags mark sections needing numbers, figures, or prose polish. Search with:

```bash
grep -R "TODO" sections/ main.tex
```

## Filling results

- Llama-3-8B repro: `tacq_data/results/REPLICATION_SUMMARY.json`
- Llama-3.1: `tacq_data/results/REPLICATION_SUMMARY_llama31.json` (when complete)
- T-DSO vs TaCQ head-to-head: `T-DSO_STATUS.md` pipeline
