
short_description: Emergency dispatch LLM bias benchmark (PPDS scale, EN/ZH)
---

# DispatchBias

An LLM bias benchmark for emergency dispatch classification. Tests whether demographic signals in a 911 call transcript shift the priority level a model assigns, holding the underlying incident constant. Eleven models, English and Mandarin Chinese, paired matched-incident scenarios.

Companion code for the paper:

LLM-DispatchBias: a cross-lingual benchmark for evaluating demographic bias in large language model emergency dispatch classification

William Guey¹, Wei Zhang¹, Pierrick Bougault¹, Yi Wang¹, Bertan Ucar¹ Vitor D. de Moura², José O. Gomes³, 

¹ Department of Industrial Engineering, Tsinghua University, Beijing, China
² School of Social Sciences, Tsinghua University, Beijing, China
³ Department of Industrial Engineering, Federal University of Rio de Janeiro, Brazil

Author contributions
W.G. conceived the study, designed and implemented the benchmark, conducted the experiments, analyzed the data, and wrote the manuscript. W.Z. provided supervision, contributed to the study design, and reviewed and edited the manuscript. P.B. contributed to scenario design and methodology refinement and edited the manuscript. Y.W contributed to methodology refinement and edited the manuscript. B.U conducted the experiments, analyzed the data.  V.D.M. contributed to social-science framing and reviewed the manuscript. J.O.G. provided supervision contributed to the methodological design and provided cross-institutional review. 



## What it does

Three steps in the UI:

1. **Import scenarios** from an Excel file. Each scenario provides a paired transcript (Variant A with a demographic signal, Variant B without) in both English and Mandarin Chinese. Only the raw transcript goes in the file. The dispatcher prompt and PPDS guide are prepended automatically at runtime.
2. **Collect data**. The app fans out async calls to OpenRouter across the selected models, with iteration-level paraphrase variation in the call openers and closers. Results land in an Excel file with one row per call.
3. **Build charts**. Five figures: per-language bias deltas, EN-vs-ZH overlay, PPDS distribution heatmap, cross-lingual scatter, and an effect size table.

## Methodology

**Scoring:** PPDS levels are scored ECHO=5, DELTA=4, BRAVO=3, ALPHA=2, OMEGA=1. Bias delta = mean PPDS(Variant A) minus mean PPDS(Variant B) across iterations. Positive deltas indicate the demographic signal raises perceived urgency, negative the opposite.

**Statistics:** Effect sizes reported as Cohen's d. Significance from independent t-tests between Variant A and Variant B score distributions per scenario, model, and language. Stars: * p<.05, ** p<.01, *** p<.001.

**Robustness:** Each prompt is paraphrased per iteration via cycling through ten matched opener-closer pairs in each language to reduce single-template artifacts.

**PPDS source:** Warner et al., *Annals of Emergency Dispatch and Response* 2014, Vol. 2 Issue 2 (IAED).

## Use your own OpenRouter key

The Space does not pay for runs. Provide your own OpenRouter API key in the field on the page. Get one at [openrouter.ai/keys](https://openrouter.ai/keys). Approximate cost: a full run (11 models, 10 scenarios, 10 iterations, 2 languages, 2 variants = 4,400 calls) typically lands under USD 5 with the default model mix.

If you fork the Space and want a default key for your own use, add `OPENROUTER_API_KEY` as a Space secret in the Settings tab.

## Local use

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-v1-..."
python app.py
```

## Citation

```bibtex
@article{guey2026dispatchbias,
  title={Emergency Dispatch LLM Bias: A Cross-Lingual PPDS Benchmark},
  author={Guey, William},
  journal={Humanities and Social Sciences Communications},
  year={2026},
  note={Under review}
}
```

## License

MIT for the code. Data and prompts are released under CC BY 4.0. The PPDS scale is the property of the IAED.
