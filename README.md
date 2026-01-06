# Reinforcement Learning from Designer Feedback - Improving User Interface Generation Models from Designer Feedback

This code repository accompanies the research paper, [Improving User Interface Generation Models from Designer Feedback](https://arxiv.org/abs/2509.16779). We release the models, designer-annotated dataset, and example inference code for the reward model and code generation models described in our publication.

Despite being trained on vast amounts of data, most LLMs are unable to reliably generate well-designed UIs. Designer feedback is essential to improving performance on UI generation; however, we find that existing RLHF methods based on ratings or rankings are not well-aligned with designers' workflows and ignore the rich rationale used to critique and improve UI designs. In this paper, we investigate several approaches for designers to give feedback to UI generation models, using familiar interactions such as commenting, sketching and direct manipulation. We first perform a study with 21 designers where they gave feedback using these interactions, which resulted in ~1500 design annotations. We then use this data to finetune a series of LLMs to generate higher quality UIs. Finally, we evaluate these models with human judges, and we find that our designer-aligned approaches outperform models trained with traditional ranking feedback and all tested baselines, including GPT-5.

## Table of Contents

- [Models](#models)
  - [Reward Model](#reward-model)
    - [Downloading Model Weights](#downloading-model-weights)
    - [Installation](#installation)
    - [Scoring a Single UI Example](#scoring-a-single-ui-example)
  - [Generation Model](#generation-model)
    - [Downloading Model Weights](#downloading-model-weights-1)
    - [Installation](#installation-1)
    - [Usage](#usage)
- [Dataset](#dataset)
  - [Download Dataset Files](#download-dataset-files)
  - [Format](#format)
- [Citation](#citation)

## Models

### Reward Model

The reward model is a fine-tuned CLIP model that scores UI quality based on visual appearance and text descriptions. It follows the standard Hugging Face model format.

### Downloading Model Weights

Download the reward model checkpoints for the different feedback conditions: [ranking](https://ml-site.cdn-apple.com/models/rldf/reward_models/rldfclip-ranking-aug.tar), [comment](https://ml-site.cdn-apple.com/models/rldf/reward_models/rldfclip-comment-aug.tar), [sketch](https://ml-site.cdn-apple.com/models/rldf/reward_models/rldfclip-sketch-aug.tar), and [revision](https://ml-site.cdn-apple.com/models/rldf/reward_models/rldfclip-revision-aug.tar).

### Installation

1. Create a new Python environment (recommended):

Using `venv`:
```bash
python -m venv venv
source venv/bin/activate
```

Or using `conda`:
```bash
conda create -n rldf python=3.10
conda activate rldf
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Scoring a Single UI Example

The `reward_model_example_inference.py` script demonstrates how to use the custom CLIP model to score a single UI screenshot.

**Usage:**

```bash
python reward_model_example_inference.py \
  --image_path /path/to/screenshot.jpg \
  --description "a modern login page with email and password fields" \
  --model_path ./rldfclip-sketch \
  --processor_path openai/clip-vit-base-patch32
```

**Required arguments:**
- `--image_path` - Path to the UI screenshot image file
- `--description` - Text description of the UI (the script automatically prepends "ui screenshot. well-designed." for scoring)

**Optional arguments:**
- `--model_path` - Path to the fine-tuned CLIP model (default: `./rldfclip-sketch`)
- `--processor_path` - Path or HuggingFace identifier for the CLIP processor (default: `openai/clip-vit-base-patch32`)
- `--dev` - Device to use: `cuda` or `cpu` (default: `cuda`)
- `--negative_weight` - Weight for negative embedding component (default: `0.5`)
- `--negative1_weight` - Weight for generic negative embedding (default: `0.9`)
- `--negative2_weight` - Weight for description-specific negative embedding (default: `0.1`)

The script outputs a numerical quality score to the console. Note that the description is automatically preprocessed with the prompting strategy used in the paper ("ui screenshot. well-designed." prefix and contrastive negative embeddings).

### Generation Model

The qwen3-coder+sketch generation model is the best performing UI code generation model from our paper's experiments. It is a Qwen3-Coder 30BA3B model fine-tuned using the sketch reward model with ORPO (Odds Ratio Preference Optimization). Keep in mind that the model was finetuned only on the specific prompt in the paper and may not generalize to other prompts and use cases.

### Downloading Model Weights

Download the [generation model weights](https://ml-site.cdn-apple.com/models/rldf/generator/qwen3-coder-sketch.gguf)

### Installation

There are many different ways of running this model which depend on available hardware and operating system. Here are installation steps using Ollama, which has cross-platform and device support. Other options for hosting and running GGUF models include llama.cpp server, vLLM, and SGLang.

1. Install Ollama:

Follow the installation instructions for your platform at [ollama.com](https://ollama.com/download)

2. Point Ollama to the model checkpoint:

Once you have the model checkpoint file, create a `Modelfile` in your working directory:

```
FROM qwen3-coder-sketch.gguf
```

Then create the model in Ollama:

```bash
ollama create rldfcoder-qwen3-sketch -f Modelfile
```

3. Run the model:

```bash
ollama run rldfcoder-qwen3-sketch
```

### Usage

Depending on the version of Ollama, the default context window might be different. You can explicitly set the context window to be greater than or equal to the value used in the paper (4096) using one of the following methods:

**Within an Ollama session:**
```
/set parameter num_ctx 4096
```

**Or by using an environment variable:**
```bash
OLLAMA_CONTEXT_LENGTH=4096 ollama run rldfcoder-qwen3-sketch
```

The `generator_model_client_inference.py` file provides an example script to query the hosted model using our target prompt.

## Dataset

The designer feedback dataset consists of about 1,460 synthetic UI screenshots generated by prompting an LLM with text descriptions and annotated by 21 professional designers through ranking, commenting, sketching, and direct revision tasks. Each interaction captured designers’ preferences and improvements, producing paired examples of better and worse designs. More collection details are available in our paper.

In this release, we split the data into four Hugging Face datasets for each of the feedback modalities.

### Download Dataset Files
Download the [dataset files](https://ml-site.cdn-apple.com/datasets/rldf/rldf.zip)

### Format
**1. Designer Comments of Synthetic UIs `comment_improved_dataset_hf`:**
- **Number of Rows:** 152
- **Column Names:** `userid`, `screenid`, `image`, `description`, `annotation`, `html_code`, `improvement_prompt`, `improved_html`, `improved_image`
    - `userid` is the participant number
    - `screenid` is the rendered UI screenshot that was labelled
    - `image` - Screenshot image of the original UI
    - `description` - Natural language description used to generate the synthetic screenshot.
    - `annotation` - Post-processed annotation from JSON file
    - `html_code` - HTML code used to generate the original UI screenshot.
    - `improvement_prompt` - Prompt for improving the original UI derived from the designer's comments.
    - `improved_html` - LLM-revised HTML code using the original HTML and improvement prompt.
    - `improved_image` - Rendered improved HTML

**2. Designer Rankings of Synthetic UIs `ranking_training_dataset_hf`:**
- **Number of Rows:** 1098
- **Column Names:** `userid`, `screenid`, `description`, `chosen_image`, `rejected_image`, `chosen_html`, `rejected_html`
    - `userid` is the participant number
    - `screenid` is the rendered UI screenshot that was labelled
    - `description` - Natural language description used to generate the synthetic screenshot.
    - `chosen_image` - From the pair of UI screenshots presented to the designer, this is the one the designer preferred.
    - `rejected_image` - From the pair of UI screenshots presented to the designer, this is the one that the designer did not choose.
    - `chosen_html` - HTML source code of the chosen image.
    - `rejected_html` - HTML source code of the rejected image.

**3. Designer Revisions of Synthetic UIs `revision_training_dataset_hf`:**
- **Number of Rows:** 68
- **Column Names:** `userid`, `description`, `chosen_image`, `rejected_image`
    - `userid` is the participant number
    - `description` - Natural language description used to generate the synthetic screenshot.
    - `chosen_image` - Rendered screenshot from designer's tweaked .sketch file.
    - `rejected_image` - Rendered screenshot from original .sketch file.

**4. Designer Comments (grounded w/ bounding boxes) of Synthetic UIs `sketch_improved_dataset_hf`:**
- **Number of Rows:** 181
- **Column Names:** `userid`, `screenid`, `image`, `description`, `annotation`, `html_code`, `improvement_prompt`, `improved_html`, `improved_image`
    - `userid` is the participant number
    - `screenid` is the rendered UI screenshot that was labelled
    - `image` - Screenshot image of the original UI
    - `description` - Natural language description used to generate the synthetic screenshot.
    - `annotation` - Post-processed annotation from JSON file
    - `html_code` - HTML code used to generate the original UI screenshot.
    - `improvement_prompt` - Prompt for improving the original UI derived from the designer's comments.
    - `improved_html` - LLM-revised HTML code using the original HTML and improvement prompt.
    - `improved_image` - Rendered improved HTML


## Citation

If you use this repository in your research, please cite the following paper:

```bibtex
@misc{wu2025improving,
  title        = {Improving User Interface Generation Models from Designer Feedback},
  author       = {Jason Wu and Amanda Swearngin and Arun Krishna Vajjala and Alan Leung and Jeffrey Nichols and Titus Barik},
  year         = {2025},
  eprint       = {2509.16779},
  archivePrefix= {arXiv},
  primaryClass = {cs.HC},
  doi          = {10.48550/arXiv.2509.16779},
  url          = {https://arxiv.org/abs/2509.16779}
}
```
