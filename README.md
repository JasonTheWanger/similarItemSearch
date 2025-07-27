# Shopee Product Matching (CLIP + FAISS)

This project implements a multi-modal product matching system for Shopee using [CLIP](https://openai.com/research/clip) embeddings and [FAISS](https://github.com/facebookresearch/faiss) for similarity search. The goal is to predict which product listings represent the same item based on both images and titles.

## Overview

Given a dataset of product listings (images and titles), this pipeline:
- Samples 3,000 items from a larger dataset.
- Extracts image embeddings using CLIP's vision encoder.
- Extracts title embeddings using CLIP's text encoder.
- Combines the image and title embeddings into a unified representation.
- Uses FAISS to perform fast nearest-neighbor search and generate predictions for matched products.
- Produces a submission file in the required Kaggle format.

## Directory Structure
```
similarItemSearch/
├── shopee-product-matching/
│ ├── train.csv # Full training data with product listings
│ ├── train_images/ # Full image dataset
│ ├── train_sample.csv # Sampled 3,000 listings metadata
│ ├── train_sample_images/ # Corresponding 3,000 image files
│ ├── image_embeddings.npy # CLIP image embeddings (3,000 x 512)
│ ├── image_title.npy # CLIP title embeddings (3,000 x 512)
│ ├── final_embeddings.npy # Combined modality embeddings (3,000 x 512)
│ ├── image_ids.txt # Posting IDs corresponding to each embedding
│ ├── sample_submission.csv # Example submission file preview
├── Embed.py # Script for extracting embeddings and combining modalities
├── FaissSearch.py # Script for building the FAISS index and generating submission
├── preprocess.py # (Optional) Script for sampling 3,000 items and copying images
└── README.md # This file
```
markdown
Copy
Edit

## Installation & Setup

1. **Clone the Repo and Create a Virtual Environment**
    ```bash
    git clone <repository_url>
    cd similarItemSearch
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. **Install Dependencies**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install git+https://github.com/openai/CLIP.git
    pip install faiss-cpu pandas numpy pillow tqdm
    pip install sentence-transformers  # if using for extended text processing
    ```

> **Note:** CLIP runs on CPU or, if available, on Apple Silicon (M1/M2/M3) via the `mps` backend.

## Process Overview

### Step 1: Sample 3,000 Items
- **Goal:** Reduce the full dataset (30k items) to a manageable 3k sample.
- **Action:** Use `preprocess.py` to randomly select 3,000 rows from `train.csv` and copy their associated images from `train_images/` into a new folder (`train_sample_images/`).

### Step 2: Extract Embeddings
- **Script:** `Embed.py`
- **Actions:**
  - Load CLIP ViT-B/32 and preprocess images.
  - For each sampled listing, extract an image embedding from the corresponding image file.
  - Tokenize and extract a title embedding from the product title (with truncation to handle token limits).
  - Combine the image and title embeddings (default is a simple average) and normalize the result.
  - Save the image embeddings, title embeddings, and combined (final) embeddings to `.npy` files.
  - Save the corresponding posting IDs to `image_ids.txt`.

### Step 3: FAISS Similarity Search and Submission Generation
- **Script:** `FaissSearch.py`
- **Actions:**
  - Load the final combined embeddings and posting IDs.
  - Build a FAISS index using cosine similarity.
  - Query the index for the top 50 most similar items for each product.
  - Optionally filter results using a similarity threshold.
  - Generate a submission CSV file (`submission.csv`) in the required format:
    - Each row contains a `posting_id` and a space-separated list of predicted matching posting IDs.

## Submission Format

The final output file, `submission.csv`, is formatted as follows:

```csv
posting_id,matches
abc123,abc123 def456 ghi789
def456,def456 abc123
```
Each row corresponds to a product.

The matches column lists the posting IDs of products predicted to be the same, including a self-match.

Running the Pipeline
(Optional) Sample the Data

bash
Copy
Edit
python preprocess.py
Extract Embeddings

bash
Copy
Edit
python Embed.py
Run FAISS Similarity Search and Generate Submission

bash
Copy
Edit
python FaissSearch.py
Future Work
Local Evaluation: Implement F1 scoring using ground truth labels from train_sample.csv.

Scaling Up: Once the pipeline is stable on 3k samples, consider running on the full dataset using GPU support on Kaggle or Colab.

Hyperparameter Tuning: Adjust weights between image and text embeddings and similarity thresholds based on local evaluation.
