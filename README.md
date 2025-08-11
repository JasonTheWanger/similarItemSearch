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
  - The final embedding is averaged from the image and the corresponding text title.
  - Save the corresponding posting IDs to `image_ids.txt`.

### Step 3: FAISS Similarity Search and Submission Generation
- **Script:** `FaissSearch.py`
- **Actions:**
  - Load the final combined embeddings and posting IDs.
  - Build a FAISS index using cosine similarity.
  - Query the index for the top 50 most similar items for each product.
  - Optionally filter results using a similarity threshold >= 0.7.
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

## Solution Comparison

This section compares different approaches for solving the Shopee Product Matching task, particularly for multi-modal (image + text) similarity. Each method is evaluated in terms of usage complexity, fine-tuning needs, performance, and speed.

### Methods Compared

| Approach | Description |
|----------|-------------|
| **CLIP + FAISS (Ours)** | Extract CLIP image and text embeddings → combine → perform nearest-neighbor retrieval using FAISS |
| **CLIP Only (Image)** | Use only CLIP's vision encoder for image-to-image similarity |
| **BERT + CNN + FAISS**  | Encode text with BERT and image with a CNN; combine embeddings and search using FAISS |
| **Sentence-BERT + CNN (Custom)** | Encode text with SBERT and image with CNN (e.g., ResNet); combine and train a similarity model |
| **Fine-tuned Siamese Network** | Train a model to distinguish matched vs. unmatched pairs using contrastive or triplet loss |
| **CLIP Fine-tuning** | Fine-tune CLIP on domain-specific image-text pairs (e.g., Shopee product matches) |

---

### Comparison Table

| Method                    | Multi-Modal | Pretrained | Fine-Tuning | Accuracy      | Speed       | Notes |
|---------------------------|-------------|------------|-------------|---------------|-------------|--------|
| **CLIP + FAISS**          | Yes         | Yes        | No          | High          | Fast        | Best zero-shot performance with easy setup |
| CLIP (Image only)         | No          | Yes        | No          | Medium        | Fast        | May miss text details like size or brand |
| **BERT + CNN + FAISS**    | Yes         | Yes        | No          | Medium-High   | Moderate    | Strong performance with more customizable components |
| **SBERT + CNN (Custom)**  | Yes         | Yes        | Optional    | Medium        |  Moderate   | Manual fusion of modalities required |
| **Fine-tuned Siamese**    | Yes         | Limited    | Yes         | Very High     | Slow        | Needs labeled pairs and more compute |
| **CLIP Fine-tuning**      | Yes         | Yes        | Yes         | Very High     | Very Slow   | High performance, high cost |

---

### Why Chose CLIP + FAISS

| Strength                | Explanation |
|-------------------------|-------------|
| **Zero-shot accuracy**  | CLIP generalizes well without retraining |
| **Cross-modal alignment** | Embeds image and title into the same space |
| **Efficient indexing**  | FAISS is fast and scalable to large datasets |
| **CPU/M1-friendly**     | Works smoothly on local machines for development |

---

### Why Use BERT + CNN

| Strength                   | Explanation |
|----------------------------|-------------|
| **Modular Design**         | BERT and CNN can be independently tuned or swapped for experimentation |
| **Strong Text Semantics**  | BERT captures rich sentence-level meaning, helpful for descriptive product titles |
| **Flexible Vision Backend**| CNNs like ResNet offer control over image representation and can be fine-tuned |
| **Complementary to CLIP**  | Useful when CLIP struggles with fine-grained domain-specific differences |

---

## Evaluation & Validation

### F1 Score Summary

We evaluated multiple versions of our model using different embedding combinations and thresholding methods:

#### Static Threshold (Threshold = 0.9)
| Version | Embedding Weights (Image / Title) | F1 Score |
|---------|-----------------------------------|----------|
| v1      | 0.5 / 0.5                         | ≈ 0.630  |
| v2      | 0.3 / 0.7                         | ≈ 0.615  |
| v3      | 0.7 / 0.3                         | ≈ 0.630  |

#### Dynamic Threshold (`threshold = mean(sim) + std(sim) * multiplier`)
| Version | Embedding Weights (Image / Title) | Scaling Multiplier | F1 Score |
|---------|-----------------------------------|---------------------|----------|
| v1      | 0.5 / 0.5                         | 1.44                | ≈ 0.690  |
| v2      | 0.3 / 0.7                         | 1.50                | ≈ 0.646  |
| v3      | 0.7 / 0.3                         | 1.45                | ≈ 0.675  |

> **Conclusion:** Dynamic thresholding yields better performance than static cutoff. The best F1 score (~0.69) is achieved using balanced embedding weights (0.5/0.5) and a dynamic threshold with a multiplier of 1.44.

#### **BERT + CNN + FAISS (Dynamic Threshold Only, `threshold = mean + std * 1.4`)**

| Version     | Embedding Weights (Image / Title) | F1 Score |
|-------------|-----------------------------------|----------|
| BERT-CNN v1 | 0.5 / 0.5                         | ≈ 0.654  |
| BERT-CNN v2 | 0.3 / 0.7                         | ≈ 0.584  |
| BERT-CNN v3 | 0.7 / 0.3                         | ≈ 0.656  |

---

## Dynamic Thresholding for Similarity Filtering

### Why Use a Dynamic Threshold?

In similarity-based retrieval (e.g., using FAISS + CLIP), using a **fixed threshold** (e.g., 0.9) for similarity score filtering can be suboptimal because:

- Some embeddings are intrinsically **closer** or **further apart** depending on how descriptive the image/title is.
- A global threshold doesn’t adapt to **local similarity variance**.
- It may include too many false positives for some queries and too few for others.

To address this, we implemented a **dynamic thresholding** mechanism that computes a tailored threshold **per query item** based on the distribution of its similarity scores.

---

### Logic Behind Dynamic Threshold

The logic is:
```python
threshold = similarities.mean() + similarities.std() * scaling_multiplier
```

---

### Recall@50 Note

While Recall@50 is commonly used in retrieval tasks, this particular Shopee dataset doesn't contain any `label_group` with more than 50 matched items. Thus, truncating predictions to 50 has **no measurable effect** on recall or F1 in this case. However, Recall@50 **would be essential** when scaling to larger datasets.

---

### Validation Method

To validate the similarity search model against ground truth:

1. **Ground Truth Dictionary Construction**:
   - The dataset contains a `label_group` column identifying all items that belong to the same group.
   - We created a dictionary that maps each `posting_id` to the set of all `posting_id`s in the same `label_group`.

2. **Prediction Dictionary**:
   - For each query item, we used FAISS to retrieve similar items (based on cosine similarity).
   - A similarity threshold was applied to retain only confident matches.

3. **F1 Score Calculation**:
   ```python
   def f1_score_row(pred_set, true_set):
       intersection = len(pred_set & true_set)
       if intersection == 0:
           return 0.0
       precision = intersection / len(pred_set)
       recall = intersection / len(true_set)
       return 2 * (precision * recall) / (precision + recall)
    ```
4. **Mean F1 Calculation**:
   ```python
   mean_f1 = sum(f1_scores) / len(f1_scores)
   ```

### Training Siamese++ Networks

This approach implements an **enhanced PxK batch Siamese network** with a **CLIP vision encoder backbone** and **SBERT text encoder fusion** for multi-modal product matching. The goal is to predict which product listings represent the same item based on both images and titles.

We started from a **CLIP + FAISS zero-shot baseline**, then moved to a **fine-tuned approach** that significantly improved F1 scores by leveraging a Siamese-style contrastive learning setup with multiple positive and negative pairs per batch (**PxK sampling**).

---

### Why Use Siamese++ (PxK Batch CLIP + SBERT Fusion)  

| Strength                       | Explanation |
|--------------------------------|-------------|
| **Multi-Modal Fine-Tuning**    | Simultaneously optimizes image and text encoders (CLIP Vision + SBERT) to align in a shared space, improving cross-modal similarity accuracy. |
| **PxK Batch Comparison**       | Processes multiple positive and negative pairs per batch, providing richer gradient signals than vanilla Siamese, leading to faster and more stable convergence. |
| **InfoNCE Loss Optimization**  | Uses CLIP-style InfoNCE loss to maximize similarity for matching pairs while minimizing it for non-matching pairs within the batch. |
| **Flexible Fusion Strategy**   | Allows adjustable weighting between image and text embeddings (e.g., 0.6 : 0.4) to adapt to dataset characteristics. |
| **Improved Performance**       | Outperforms frozen CLIP + FAISS and vanilla Siamese by leveraging domain-specific fine-tuning, yielding ~6% F1 improvement in our case. |

---

## Core Ideas

### 1. **CLIP Backbone Fine-Tuning**
- Use CLIP’s vision encoder for images.
- Fine-tune only the **image encoder** during PxK Siamese training (title encoder is from SBERT for fusion stage).

### 2. **PxK Batch Siamese Training**
- Each batch contains **P labels × K samples per label**.
- Every sample forms **both positive and negative pairs** with others in the batch.
- Contrastive loss: **CLIP-style InfoNCE**.
- Compared to vanilla Siamese (1 pos / 1 neg per step), PxK allows **many comparisons at once**, improving training efficiency and stability.

### 3. **SBERT Fusion at Inference**
- Since SBERT is already a pretrained model from Siamese + BERT, we will just take that into use.
- At test time, fuse CLIP image embeddings with SBERT text embeddings:
  ```
  final_embedding = img_emb * α + text_emb * (1 − α)
  ```
- Best performance so far at **α = 0.6** (image weight).

---

## Outcome

**Dataset Split:**  
- Train: 70%  
- Val: 15%  
- Test: 15%

**Image-only model:**  
- Best epoch: 8  
- Val F1: **0.7375** @ threshold 0.56  
- Test F1 (val threshold): **0.6878**  
- Test F1 (best test threshold 0.54): **0.6960**  

**SBERT fusion (α = 0.6, epoch 8):**  
- Val F1: **0.7939**  
- Test F1: **0.7566**  

**Improvement:**  
- Nearly **+6% absolute F1** vs. frozen CLIP or BERT+CNN baseline.

---

## What Didn’t Work

- **Vanilla Siamese:**  
  - Using a 1:1 pos/neg pair training loop with CLIP embeddings did not converge as well.
  - Low diversity in comparisons per step → slower learning.
  - PxK sampling + CLIP-style InfoNCE yielded far better results.

---

## Directory Structure
```
similarItemSearch/
├── train_clip.py            # PxK batch Siamese training
├── two_pair_fusion_infer.py # Inference with image+text fusion on two data input and determine if similar or not
├── datasets/                # Data loading & transforms
├── checkpoints/             # Saved model weights
├── train.csv                # Metadata
├── train_images/            # Image data
└── README.md                # This file
```

---

## How to Run

### 1) Train + Validate + Test (image-only)

Runs PxK training, saves best checkpoint by VAL F1, and evaluates on TEST with the VAL-chosen threshold.
```bash
python train_clip.py --epochs 8 --batch 64 --P 16 --K 4
```

### 2) Train + Validate + Test (with SBERT fusion)

Same as above, plus builds SBERT embeddings after training, grid-searches the image weight on VAL, and reports fused F1 on TEST.
```bash
python train_clip.py \
  --epochs 8 --batch 64 --P 16 --K 4 \
  --fuse --sbert_model all-MiniLM-L6-v2 \
  --wimg_grid 0.60,0.70,0.80,0.85,0.90,0.95
```

---

## Evaluation

- F1 scores are computed on **validation split** to select the best epoch & threshold.
- Test scores are reported with both:
  - Val-tuned threshold (**fair**)
  - Test-tuned threshold (**upper bound** for debugging)

---

## Original CLIP + FAISS Baseline

Before Siamese++ fine-tuning, we used a zero-shot CLIP + FAISS pipeline:

### Step 1: Extract Embeddings
- CLIP ViT-B/32 for image & text embeddings.
- Normalize and combine (average) image & title embeddings.

### Step 2: Similarity Search
- FAISS index with cosine similarity.
- Retrieve top-K matches per item.

### Step 3: Thresholding
- Dynamic threshold per query: `mean(sim) + std(sim) * multiplier`.

---

## Future Work
- Larger-scale training with full dataset.
- More advanced fusion strategies (cross-attention, gating networks).
- Domain-specific CLIP pretraining.
- Cross-modal supervision, include fine tuning SBERT in the P x K Siamese++ training process.



