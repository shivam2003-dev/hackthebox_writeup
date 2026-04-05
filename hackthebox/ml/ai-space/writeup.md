# HackTheBox — AI Space | Writeup

**Category:** AI / ML  
**Difficulty:** Easy  
**Flag:** `HTB{d1st4nt_spac3}`

---

## Table of Contents

1. [Challenge Description](#challenge-description)
2. [Initial Recon](#initial-recon)
3. [Core Concepts You Need to Know](#core-concepts-you-need-to-know)
   - [What is a Distance Matrix?](#what-is-a-distance-matrix)
   - [What is Dimensionality Reduction?](#what-is-dimensionality-reduction)
   - [What is MDS (Multidimensional Scaling)?](#what-is-mds-multidimensional-scaling)
   - [MDS vs t-SNE vs PCA — When to Use Which](#mds-vs-t-sne-vs-pca--when-to-use-which)
4. [Solving the Challenge](#solving-the-challenge)
5. [Deep Dive: How the Challenge Was Built](#deep-dive-how-the-challenge-was-built)
6. [Exploring Alternatives](#exploring-alternatives)
7. [Takeaways for AI/ML CTFs](#takeaways-for-aiml-ctfs)

---

## Challenge Description

> You are assigned the important mission of locating and identifying the infamous space hacker. Your investigation begins by analyzing the data patterns and breach points identified in the latest cyber-attacks. Use the provided coordinates of the last known signal origins to narrow down his potential hideouts. Utilize advanced tracking algorithms to follow the digital footprint left by the hacker.

**Given file:** `distance_matrix.npy`

The key phrases in the description are deliberate hints:
- **"coordinates"** → spatial/geometric data
- **"signal origins"** → points in some space
- **"advanced tracking algorithms"** → dimensionality reduction / manifold learning
- **"digital footprint"** → the pattern will be visually revealed

---

## Initial Recon

The first step in any data challenge is understanding what you have. Always inspect before you analyze.

```python
import numpy as np

data = np.load('distance_matrix.npy')

print("Shape :", data.shape)    # (1808, 1808)
print("Dtype :", data.dtype)    # float64
print("Min   :", data.min())    # 0.0
print("Max   :", data.max())    # ~7.21

# Check symmetry: a valid distance matrix must be symmetric
print("Symmetric:", np.allclose(data, data.T))  # True

# Check diagonal: distance from a point to itself must be 0
print("Zero diagonal:", np.allclose(np.diag(data), 0))  # True

# Check a few pairwise distances
print(data[0, 1])   # 0.436  (points 0 and 1 are close)
print(data[0, -1])  # 7.206  (points 0 and 1807 are far)
print(data[-1, -2]) # 0.013  (points 1806 and 1807 are close)
```

**What this tells us:**
- We have **1808 entities** with pairwise distances computed between all pairs
- The matrix is **symmetric** (distance A→B = distance B→A) ✓
- The **diagonal is zero** (distance of a point to itself) ✓
- Points at the start (low indices) are close to each other
- Points at the end (high indices) are close to each other  
- Start and end points are far apart (~7.2 units)
- This suggests the points form some kind of **structured geometric arrangement**

---

## Core Concepts You Need to Know

### What is a Distance Matrix?

A **distance matrix** `D` is an `n × n` matrix where `D[i][j]` represents the distance between point `i` and point `j`.

```
       P0    P1    P2    P3
P0  [  0.0   0.44  0.36  7.20 ]
P1  [  0.44  0.0   0.08  7.19 ]
P2  [  0.36  0.08  0.0   7.19 ]
P3  [  7.20  7.19  7.19  0.0  ]
```

Properties of a valid distance matrix:
1. **Zero diagonal**: `D[i][i] = 0` (a point has zero distance to itself)
2. **Symmetry**: `D[i][j] = D[j][i]` (distance is the same in both directions)
3. **Non-negativity**: `D[i][j] >= 0` (distances can't be negative)
4. **Triangle inequality**: `D[i][k] <= D[i][j] + D[j][k]`

**Common uses in ML:**
- Clustering algorithms (DBSCAN, hierarchical clustering)
- Nearest-neighbor search
- Manifold learning / dimensionality reduction
- Graph-based algorithms

**Common distance metrics:**
| Metric | Formula | Use case |
|--------|---------|---------|
| Euclidean | `sqrt(sum((a-b)²))` | Geometric/spatial data |
| Manhattan | `sum(\|a-b\|)` | Grid-like spaces |
| Cosine | `1 - (a·b)/(‖a‖‖b‖)` | Text/NLP similarity |
| Hamming | Count of differing bits | Binary/categorical data |

---

### What is Dimensionality Reduction?

In machine learning, data often lives in **high-dimensional space** (hundreds or thousands of features). Humans can only visualize up to 3 dimensions. Dimensionality reduction techniques **compress** high-dimensional data into 2D or 3D while preserving important structure.

```
High-dimensional space         2D projection
  (n dimensions)                  (2D plot)
  
  [0.2, 0.8, 0.1, ...]  →→→    (x, y)
  [0.5, 0.3, 0.9, ...]  →→→    (x, y)
  [0.1, 0.7, 0.4, ...]  →→→    (x, y)
```

**Why it matters:**
- Visualization of complex data
- Noise reduction
- Feature extraction
- Anomaly detection

The key challenge: how do you decide _which_ information to preserve when compressing? Different algorithms make different trade-offs.

---

### What is MDS (Multidimensional Scaling)?

**MDS** is a dimensionality reduction technique that works directly from a **distance matrix** — you don't need the original data points, just the pairwise distances between them.

**Goal:** Find a set of 2D (or 3D) coordinates such that the Euclidean distances between the projected points are as close as possible to the original distances.

**Intuition:** Imagine you have a table of driving distances between 10 cities, but no map. MDS can reconstruct a map that approximately matches all those distances.

```
Input: Distance matrix D (n×n)

           NYC  LA   CHI  HOU
NYC      [  0  2790 790  1630]
LA       [2790   0  2020  1550]
CHI      [ 790 2020   0   1090]
HOU      [1630 1550 1090    0 ]

Output: 2D coordinates
  NYC → (0.3, 0.8)
  LA  → (-0.9, -0.1)
  CHI → (0.1, 0.4)
  HOU → (-0.2, -0.3)
       → Plot these → looks like a map of the USA!
```

**How MDS works mathematically:**

MDS minimizes a cost function called **stress**:

```
Stress = sqrt( sum_ij (d_ij - δ_ij)² / sum_ij δ_ij² )
```

Where:
- `d_ij` = distance between points i and j in the **original** space
- `δ_ij` = Euclidean distance between projected points i and j in **2D**

Lower stress = better reconstruction. Stress = 0 means perfect reconstruction.

**In sklearn:**

```python
from sklearn.manifold import MDS

# When you have raw data (sklearn computes distances internally)
mds = MDS(n_components=2)
X_2d = mds.fit_transform(X_raw)

# When you already have a distance matrix (our case)
mds = MDS(n_components=2, dissimilarity='precomputed')
X_2d = mds.fit_transform(distance_matrix)
#                         ^^^^^^^^^^^^^^
#                         Pass the matrix directly — sklearn won't recompute distances
```

The `dissimilarity='precomputed'` parameter is critical — it tells sklearn: "I'm passing you distances, not raw features."

---

### MDS vs t-SNE vs PCA — When to Use Which

These are the three most common dimensionality reduction techniques you'll encounter in AI/ML CTFs and real ML work.

| Feature | PCA | MDS | t-SNE |
|---------|-----|-----|-------|
| **Input** | Raw data matrix | Raw data or distance matrix | Raw data or distance matrix |
| **Preserves** | Global variance / linear structure | Global distances | Local neighborhood structure |
| **Deterministic** | Yes | No (random init, use `random_state`) | No (random init, use `random_state`) |
| **Scalability** | Fast (O(n²)) | Slow (O(n³)) | Medium (O(n² log n) with BH) |
| **Good for** | Linear relationships, feature importance | Recovering geometric layout | Cluster visualization |
| **Precomputed matrix** | No (needs raw features) | Yes ✓ | Yes ✓ |

**Rule of thumb for CTFs:**
- Have a **distance matrix**? → Try **MDS** first, then **t-SNE**
- Have **raw features**? → Try **PCA** first for speed, then t-SNE for cluster visualization
- Want to **recover original geometry**? → **MDS** is the right tool

```python
# PCA — needs raw data, finds linear components
from sklearn.decomposition import PCA
X_2d = PCA(n_components=2).fit_transform(X_raw)

# MDS — works with distance matrices, preserves global distances
from sklearn.manifold import MDS
X_2d = MDS(n_components=2, dissimilarity='precomputed').fit_transform(dist_matrix)

# t-SNE — works with distance matrices, great for cluster visualization
from sklearn.manifold import TSNE
X_2d = TSNE(n_components=2, metric='precomputed').fit_transform(dist_matrix)
```

---

## Solving the Challenge

With the theory in place, the solution is clean:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Step 1: Load the distance matrix
data = np.load('distance_matrix.npy')
# Shape: (1808, 1808) — 1808 points with all pairwise distances

# Step 2: Apply MDS
# - n_components=2: reduce to 2D for visualization
# - dissimilarity='precomputed': input is already a distance matrix
# - random_state=42: fix the random seed for reproducibility
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
X = mds.fit_transform(data)
# X.shape → (1808, 2): each of the 1808 points now has (x, y) coordinates

# Step 3: Plot the reconstructed 2D coordinates
plt.figure(figsize=(16, 8))
plt.scatter(X[:, 0], X[:, 1], s=5, c='cyan', marker='.')
plt.axis('off')
plt.tight_layout()
plt.savefig('flag.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Output

When plotted, the 1808 dots arrange themselves to spell out the flag:

```
HTB{d1st4nt_spac3}
```

The flag itself is a leet-speak hint at the technique: **"distant space"** → the data was arranged in distant high-dimensional space, recoverable via MDS.

---

## Deep Dive: How the Challenge Was Built

Understanding how the challenge was _created_ deepens your understanding of the math.

**Step 1 — Design the text as 2D points**

The challenge creator started with 2D coordinates `(x, y)` for 1808 points arranged to spell `HTB{d1st4nt_spac3}`. Each dot in the text is one point.

```python
# Conceptually (challenge creator's perspective):
# Place dots along character outlines to spell the flag
points_2d = generate_text_as_points("HTB{d1st4nt_spac3}", n_points=1808)
# points_2d.shape → (1808, 2)
```

**Step 2 — Compute the pairwise distance matrix**

```python
from sklearn.metrics import pairwise_distances

# Compute all 1808×1808 pairwise Euclidean distances
distance_matrix = pairwise_distances(points_2d, metric='euclidean')
# distance_matrix.shape → (1808, 1808)
# This is exactly what was saved to distance_matrix.npy
```

**Step 3 — Save and distribute**

```python
np.save('distance_matrix.npy', distance_matrix)
```

The original coordinates are **discarded** — only the distance matrix is given to the player. The challenge is to reverse this: recover the coordinates from only the distances.

**Why MDS can reverse this perfectly:**

Because the points were originally in 2D Euclidean space, MDS can recover them exactly (up to rotation/reflection/translation — the absolute position doesn't matter, only relative distances). This is why `random_state=42` matters — without fixing the seed, the reconstruction might be rotated or flipped differently each run.

```
Original 2D coords  →  Distance matrix  →  MDS  →  Recovered 2D coords
   (discarded)          (given to you)              (≈ original, may be rotated)
```

---

## Exploring Alternatives

### Alternative 1: t-SNE

t-SNE is another popular dimensionality reduction technique that also accepts a precomputed distance matrix:

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, metric='precomputed', random_state=42, 
            perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(data)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5)
plt.show()
```

**Would t-SNE work here?** Possibly, but with caveats:
- t-SNE is better at preserving **local** structure (clusters) than global geometry
- The text characters might appear as blobs rather than readable letters
- MDS is the better choice when you want to faithfully recover the original layout

### Alternative 2: DBSCAN (Anomaly Detection)

Before landing on MDS, one might try clustering to find an "outlier hacker":

```python
from sklearn.cluster import DBSCAN

# DBSCAN with precomputed distance matrix
db = DBSCAN(eps=0.5, min_samples=5, metric='precomputed')
labels = db.fit_predict(data)

# Noise points (label = -1) are outliers
noise_indices = np.where(labels == -1)[0]
print("Outlier indices:", noise_indices)
```

This approach would find isolated points but wouldn't reveal the visual flag.

### Alternative 3: Nearest Neighbor Analysis

```python
# For each point, find its nearest neighbor distance
# (minimum non-zero distance in each row)
nn_distances = []
for i in range(len(data)):
    row = data[i].copy()
    row[i] = np.inf  # exclude self
    nn_distances.append(row.min())

nn_distances = np.array(nn_distances)
print("Most isolated point:", np.argmax(nn_distances))
print("Max NN distance:", nn_distances.max())
```

This might help find anomalies but again doesn't reveal the flag visually.

**Conclusion:** MDS is uniquely suited here because the challenge specifically encodes information in the **global geometric layout** of the points.

---

## Takeaways for AI/ML CTFs

### 1. Recognize Data Types Immediately

| File type | What it likely contains | First tools to try |
|-----------|------------------------|-------------------|
| `.npy` | NumPy array — could be images, embeddings, distance matrix | `np.load()`, check shape/dtype |
| `.pkl` | Serialized Python object — model, dataset, dict | `pickle.load()` |
| `.h5` / `.hdf5` | Neural network weights or large datasets | `h5py` or `keras.models.load_model()` |
| `.pt` / `.pth` | PyTorch model or tensor | `torch.load()` |
| `.csv` | Tabular data | `pandas.read_csv()` |

### 2. When You See a Distance/Similarity Matrix

Immediately ask:
- Is it **symmetric**? → valid distance matrix
- Is the diagonal **zero**? → confirms it's a distance matrix (not similarity)
- What's the **value range**? → helps choose epsilon for DBSCAN, etc.

Then try in order:
1. **MDS** — recovers global geometric layout, may spell text or reveal patterns
2. **t-SNE** — reveals cluster structure
3. **DBSCAN** — finds outliers and dense clusters
4. **Hierarchical clustering** — builds a dendrogram showing group structure

### 3. Always Fix Random Seeds

Many dimensionality reduction techniques are non-deterministic (random initialization):

```python
# Always use random_state for reproducibility
mds  = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
tsne = TSNE(n_components=2, metric='precomputed',       random_state=42)
```

Without a fixed seed, the output might be rotated, flipped, or in a completely different orientation on each run.

### 4. Visualize Everything

In AI/ML CTFs, the flag is often **hidden visually** — in a scatter plot, a heatmap, an image reconstructed from embeddings, etc. Always plot your data:

```python
# Scatter plot for 2D point clouds
plt.scatter(X[:, 0], X[:, 1], s=5)

# Heatmap for matrices
plt.imshow(matrix, cmap='viridis')
plt.colorbar()

# Image reconstruction from flattened arrays
image = array.reshape(height, width)
plt.imshow(image, cmap='gray')
```

### 5. Read the Filename and Challenge Title

- `distance_matrix.npy` → immediately signals a precomputed distance matrix
- "AI Space" → spatial/geometric data, coordinates, layout
- These clues narrow the solution space before you write a single line of code

---

## Summary

```
distance_matrix.npy
        │
        ▼
   Load with numpy
   Shape: (1808, 1808)
   Values: 0.0 → 7.21
        │
        ▼
   Apply MDS
   dissimilarity='precomputed'
   n_components=2
        │
        ▼
   1808 (x, y) coordinates
        │
        ▼
   Scatter plot
        │
        ▼
   Flag: HTB{d1st4nt_spac3}
```

The challenge elegantly demonstrates that a distance matrix contains rich structural information — and MDS is the key to unlocking it.

**Flag: `HTB{d1st4nt_spac3}`**
