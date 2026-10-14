import json, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

R, SEEDS = Path("results"), [0, 1, 2]
names = json.load(open(R / "seed0/summary.json"))["class_names"]
si, fi = names.index("steak"), names.index("filet_mignon")

# 1. Boundary shift check: did fine-tuning just move steak <-> filet predictions?
for c in ["control", "finetune"]:
    tot = sum(confusion_matrix(np.load(R / f"seed{s}/test_labels.npy"),
                               np.load(R / f"seed{s}/{c}_test_probs.npy").argmax(1)) for s in SEEDS)
    print(f"{c:9s} steak->filet {tot[si, fi]:4d}   filet->steak {tot[fi, si]:4d}   "
          f"predicted-as-filet total {tot[:, fi].sum():4d}")

# 2. Look at the actual errors: steak images the model called filet_mignon (seed 0)
ds = tf.keras.utils.image_dataset_from_directory("data/test", image_size=(224, 224),
                                                 batch_size=32, shuffle=False)
paths = ds.file_paths
y = np.load(R / "seed0/test_labels.npy")
p = np.load(R / "seed0/finetune_test_probs.npy").argmax(1)
idx = np.where((y == si) & (p == fi))[0][:16]
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for ax, i in zip(axes.flat, idx):
    ax.imshow(plt.imread(paths[i])); ax.set_title(Path(paths[i]).name, fontsize=8); ax.axis("off")
plt.suptitle("True: steak   Predicted: filet_mignon   (fine-tuned, seed 0)")
plt.tight_layout(); plt.savefig(R / "errors_steak_as_filet.png", dpi=120)