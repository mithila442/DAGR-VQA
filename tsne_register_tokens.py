import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from collections import Counter

# Load data
data = np.load("token_embeddings_epoch_170.npz", allow_pickle=True)
tokens = data["tokens"]
labels = data["labels"]

print(f"Loaded {len(tokens)} embeddings.")
print("Original Label distribution:", Counter(labels))

# === Remap labels to 4 categories and exclude 'unknown' ===
def remap_label(label):
    label = str(label)
    if label in ["Social activity", "Sport", "Artistic performance"]:
        return "People"
    elif label == "Animal":
        return "Animal"
    elif label == "Landscape":
        return "Nature"
    elif label == "Artifact":
        return "Object"
    else:
        return None  # exclude 'unknown' or any unmapped

# Apply remapping and filter
filtered_tokens, filtered_labels = [], []
for tok, lab in zip(tokens, labels):
    new_label = remap_label(lab)
    if new_label is not None:
        filtered_tokens.append(tok)
        filtered_labels.append(new_label)

tokens = np.array(filtered_tokens)
labels = np.array(filtered_labels)

print("Filtered Label distribution:", Counter(labels))

# Mean pool if needed
if len(tokens.shape) == 3:
    tokens = tokens.mean(axis=1)

# Normalize token vectors to prevent t-SNE instability
tokens = (tokens - tokens.mean(axis=0)) / (tokens.std(axis=0) + 1e-8)
tokens = np.nan_to_num(tokens)

# Debug: check token stats
print("Token stats after normalization:")
print("  Mean:", tokens.mean())
print("  Std:", tokens.std())
print("  Max:", tokens.max())
print("  Min:", tokens.min())

# Run t-SNE
print("Running t-SNE (no PCA)...")
proj = TSNE(
    n_components=2,
    perplexity=17,
    init="random",
    random_state=42
).fit_transform(tokens)

# =========================
# Plotting (LEGEND INSIDE, OVAL PRESERVED)
# =========================
fig, ax = plt.subplots(figsize=(12, 7))

unique_labels = sorted(set(labels))
palette = sns.color_palette("Dark2", n_colors=len(unique_labels))
label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

# Plot per class
for label in unique_labels:
    idx = labels == label
    ax.scatter(
        proj[idx, 0], proj[idx, 1],
        color=label_to_color[label],
        s=180,
        alpha=0.7,
        edgecolors='k',
        linewidths=0.5,
        label=label
    )

# Fonts
TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 15
LEGEND_TITLE_FONTSIZE = 17

ax.set_title("t-SNE of Register Token Embeddings", fontsize=TITLE_FONTSIZE, fontweight='bold')
ax.set_xlabel("Dimension 1", fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
ax.set_ylabel("Dimension 2", fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)

# ---- Key part: create space for legend INSIDE ----
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin-0.25, ymax + 0.40)   # small upward padding only

# Legend inside, readable, non-obstructive
legend = ax.legend(
    loc="upper right",
    fontsize=LEGEND_FONTSIZE,
    title_fontsize=LEGEND_TITLE_FONTSIZE,
    frameon=True,
    markerscale=1.3,
    borderpad=0.8,
    labelspacing=0.6,
    handletextpad=0.6
)

# Slight transparency so points beneath are still visible
legend.get_frame().set_alpha(0.9)

# Make legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')

legend.get_title().set_fontweight('bold')

plt.tight_layout()
plt.savefig("tsne_register_tokens.png", dpi=600, bbox_inches="tight")
plt.show()