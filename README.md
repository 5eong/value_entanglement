# 2025_value_entanglement

Code, stimuli, and data for a project on value entanglement in large language
models: the finding that a model's judgments of one kind of value, such as
grammaticality or economic worth, leak into its judgments of moral value.

exp3 measures the effect behaviorally, by asking hosted models to rate the same
stimuli on two dimensions at a time. exp4 looks for it in the residual stream of
open-weight models. exp5 intervenes on the directions exp4 finds. The notebooks in
`controls/` contains the control experiments.

## Setup

This repository does not use Portobello. The experiments run on
a local Python install, because exp4 and exp5 need the host machine's GPU.

1. Check out the folder and open it in VSCode.
2. Install the Python packages:
   ```console
   pip install -r requirements.txt
   ```
3. Copy `.env.template` to `.env` and fill in the keys you need. `.env` is never
   committed.

   | Key | Used by |
   |---|---|
   | `OPENROUTER_API_KEY` | exp3 and every notebook in `controls/` |
   | `ANTHROPIC_API_KEY` | the judge in `grammar_corruption.ipynb` and `valence_controls.ipynb` |
   | `OPENAI_API_KEY` | direct OpenAI calls, where a notebook bypasses OpenRouter |
   | `DEEPINFRA_API_KEY`, `FEATHERLESS_API_KEY` | second source for models OpenRouter has withdrawn |
   | `VAST_API_KEY` | renting a GPU through `src/vast.py` for a model no provider serves |
   | `HF_TOKEN` | downloading open-weight models for exp4 and exp5 |
   | `GH_TOKEN` | optional, for `gh` CLI work in this repository |

   Certain models are region locked, and I'm (Seong) in Hong Kong. Where the key
   billed to one account is refused, swap `OPENROUTER_API_KEY` in `.env` for a key
   on an account that is not: there is one key per provider and no automatic
   fallback.
4. exp4 and exp5 need a CUDA GPU. exp3 and the control analyses run anywhere.

## Getting started

Open [`exp3/plotting.ipynb`](exp3/plotting.ipynb) and run it. It
reads the ratings in `exp3/data/` and reproduces the figures without calling any
API. Collection requires OpenRouter API key. `exp3/behavior.ipynb` starts querying OpenRouter as soon as it is run, so edit its model list first. The notebooks in `controls/` are gated
instead, and cache what they collect, so an interrupted run resumes where it
stopped:

Notebooks are run from their own folder, so stimulus paths appear as `../stimuli/`
and the first cell puts `src/` on `sys.path`.

## Adding packages

### Python packages

Add the package to [`requirements.txt`](requirements.txt), then install:

```console
pip install -r requirements.txt
```

### Other tools

There is no container to rebuild, so anything outside Python is installed on the
machine itself: the NVIDIA driver and CUDA runtime that `torch` binds to for exp4
and exp5. Record any further system dependency here, with the version it was
tested against.

exp5 clears each open-weight model from the Hugging Face cache before downloading
the next to save disk space.

## Where are my files?

| What is it?           | Where       | Notes                                                    |
|-----------------------|-------------|----------------------------------------------------------|
| Shared library        | `src/`      | Imported by every experiment.                             |
| Stimuli               | `stimuli/`  | The rated item sets; `stimuli/controls/` for the controls. |
| Behavioral experiment | `exp3/`     | Ratings from hosted models.                               |
| Activations           | `exp4/`     | Residual stream activations and concept projections.      |
| Interventions         | `exp5/`     | Steering and direction ablation.                          |
| Control analyses      | `controls/` | Tests of alternative accounts, and the model selection they share. |

Each experiment folder holds its notebook or script, a `data/` directory with one
subdirectory per model, and a `figures/` directory. Where a folder has both a
collection notebook and a `plotting.ipynb`, the first produces the data and the
second reads it back.

```
src/         activation.py  env_keys.py  intervention.py  load_data.py  prompts.py
             vast.py  query_model/__init__.py  query_model/model_routing.py
             query_model/generation.py
stimuli/     MoralGrammar68.csv  MoralEconomic68.csv  Dillion_et_al_2023.csv
             moralgrammar68_human_ratings.json  Grand_et_al/  controls/
exp3/        behavior.ipynb  plotting.ipynb  data/<model>/  figures/
exp4/        activations.ipynb  plotting.ipynb  data/<model>/  figures/
exp5/        experiments.py  plotting.ipynb  data/<model>/  figures/
controls/    controls.py  prompting_variation.ipynb  grammar_corruption.ipynb
             valence_controls.ipynb  probe/  data/
```

## Project notes

`query_model/` runs hosted models: `model_routing.py` picks a provider, `generation.py` sends the request, and parses the replies and runs the rating protocol. `prompts.py` holds every prompt and chat template, so a prompting changes only happens once. `activation.py` builds one direction per layer from a difference of means, `intervention.py` runs ablation and steering, `load_data.py` reads the stimulus files, `env_keys.py` loads the keys, and `vast.py` rents a GPU when no hosted provider serves a model. `controls/controls.py` holds the model selection the control analyses share, beside the notebooks that import it.

**exp3** has each model rate a random 10-item subset per iteration, 100 iterations
per model and dimension. The entanglement statistic is the Pearson correlation
between a model's `morality_mean` and its secondary-dimension mean across the 68
stimuli; humans rating the same items correlate at r = .05. **exp4** extracts a
direction per layer for each concept and projects the stimuli onto it, with the
Grand et al. sets as dimensions that carry no value. **exp5** sweeps interventions
over a steering concept, an evaluation dataset, a type, and a strength.

`MoralGrammar68.csv` crosses 17 scenarios with 4 levels of syntactic corruption,
level 1 uncorrupted, and `MoralEconomic68.csv` embeds a purchasable good in each of
the same scenarios at its real retail price. `Dillion_et_al_2023.csv` is the human
benchmark, and `stimuli/controls/` holds the valence, AntiValence and negated
variants.

Selection in `controls/` is per stimulus set: of the 31 exp3 models, 11 were
significant on moral by grammatical at alpha = .05, 19 on moral by economic, and ten
on both. `valence_controls.ipynb` asks whether the effect is about morality or only
about affect, `grammar_corruption.ipynb` whether the corruption changed meaning
rather than form, and `prompting_variation.ipynb` whether the correlations survive
other prompt wordings.
