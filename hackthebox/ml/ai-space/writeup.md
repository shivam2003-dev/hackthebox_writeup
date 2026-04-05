# HackTheBox — AI Space | Writeup

**Category:** AI / ML  
**Difficulty:** Easy  
**Flag:** `HTB{d1st4nt_spac3}`

---

## Challenge Description

> You are assigned the important mission of locating and identifying the infamous space hacker. Your investigation begins by analyzing the data patterns and breach points identified in the latest cyber-attacks. Use the provided coordinates of the last known signal origins to narrow down his potential hideouts. Utilize advanced tracking algorithms to follow the digital footprint left by the hacker.

**Given file:** `distance_matrix.npy`

---

## Initial Recon

Load and inspect the file:

```python
import numpy as np

data = np.load('distance_matrix.npy')
print(data.shape)   # (1808, 1808)
print(data.dtype)   # float64
print(data.min(), data.max())  # 0.0  ~7.21
```

Key observations:
- **1808 × 1808 symmetric matrix** — pairwise distances between 1808 points
- Diagonal is all zeros (distance from a point to itself)
- Values range from `0.0` to `~7.21`
- Points near index 0 are close to each other; points near index 1807 are also close to each other; but the two ends are ~7.2 apart — suggesting a structured layout in some high-dimensional space

---

## Key Insight

The challenge says **"coordinates of last known signal origins"** and **"advanced tracking algorithms"**. The filename itself is `distance_matrix.npy` — a precomputed pairwise distance matrix.

This screams **Multidimensional Scaling (MDS)**: a technique that takes a distance matrix and reconstructs the original coordinates in a lower-dimensional space (2D or 3D) while preserving relative distances.

The hypothesis: the 1808 points were originally arranged to **spell out a message** in some high-dimensional space. MDS will "unfold" that space back into 2D, revealing the text visually.

---

## Solution

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Load the precomputed distance matrix
data = np.load('distance_matrix.npy')

# Apply MDS — dissimilarity='precomputed' tells sklearn the matrix is already distances
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
X = mds.fit_transform(data)  # Shape: (1808, 2)

# Plot the reconstructed 2D coordinates
plt.figure(figsize=(16, 8))
plt.scatter(X[:, 0], X[:, 1], s=5, c='cyan', marker='.')
plt.axis('off')
plt.tight_layout()
plt.savefig('flag.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Output

The scatter plot reveals the flag spelled out in dots:

```
HTB{d1st4nt_spac3}
```

The 1808 points were embedded in a high-dimensional space with their pairwise distances engineered so that MDS reconstruction maps them onto text characters.

---

## Why MDS Works Here

| Step | What happens |
|------|-------------|
| Input | 1808×1808 distance matrix |
| MDS | Finds 2D coordinates that best preserve all pairwise distances |
| Output | 1808 (x, y) points that, when plotted, form readable text |

MDS minimizes **stress** — the difference between original distances and Euclidean distances in the reduced space. Because the points were _designed_ to spell text in a 2D plane, MDS recovers those coordinates nearly perfectly.

---

## Tools Used

- `numpy` — load `.npy` file
- `sklearn.manifold.MDS` — dimensionality reduction with precomputed distance matrix
- `matplotlib` — scatter plot visualization

---

## Takeaway

When you see a **precomputed distance/dissimilarity matrix** in an AI/ML CTF:
1. Try **MDS** with `dissimilarity='precomputed'` — it may visually reveal a hidden message
2. Try **TSNE** with `metric='precomputed'` as an alternative
3. The points are often engineered to spell text or form a recognizable pattern in 2D

**Flag: `HTB{d1st4nt_spac3}`**
