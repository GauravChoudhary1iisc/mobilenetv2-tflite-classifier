import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

R, SEEDS = Path("results"), [0, 1, 2]
S = [json.load(open(R / f"seed{s}/summary.json")) for s in SEEDS]
names = S[0]["class_names"]

# 1. Summary table across seeds
for k in ["stage1_frozen_10ep", "control_frozen_20ep", "finetune_20ep"]:
    v = np.array([s[k] for s in S])
    print(f"{k:22s} mean {v.mean():.4f}  std {v.std(ddof=1):.4f}  per-seed {v.round(4)}")
gain = np.array([s["finetune_20ep"] - s["control_frozen_20ep"] for s in S])
print("fine-tune gain over control, per seed:", gain.round(4), " mean:", round(gain.mean(), 4))

# 2. Training curves (seed 0): stage 1, then control vs fine-tune
h = {n: json.load(open(R / f"seed0/{n}_history.json")) for n in ["stage1", "control", "finetune"]}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
e1, e2 = range(1, 11), range(11, 21)
for i, m in enumerate(["accuracy", "loss"]):
    ax[i].plot(e1, h["stage1"][m], "k-", label="stage 1 train")
    ax[i].plot(e1, h["stage1"]["val_" + m], "k--", label="stage 1 val")
    ax[i].plot(e2, h["control"][m], "C0-", label="control (frozen) train")
    ax[i].plot(e2, h["control"]["val_" + m], "C0--", label="control (frozen) val")
    ax[i].plot(e2, h["finetune"][m], "C3-", label="fine-tune train")
    ax[i].plot(e2, h["finetune"]["val_" + m], "C3--", label="fine-tune val")
    ax[i].axvline(10.5, color="gray", ls=":")
    ax[i].set_xlabel("epoch"); ax[i].set_title(m); ax[i].grid(alpha=.3)
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(R / "curves_seed0.png", dpi=150)

# 3. Confusion matrix (fine-tuned, seed 0), row-normalized
y = np.load(R / "seed0/test_labels.npy")
p = np.load(R / "seed0/finetune_test_probs.npy").argmax(1)
cm = confusion_matrix(y, p, normalize="true")
fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_yticks(range(10)); ax.set_yticklabels(names)
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                color="white" if cm[i, j] > .5 else "black", fontsize=8)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Fine-tuned MobileNetV2, test set (seed 0)")
plt.tight_layout(); plt.savefig(R / "confusion_finetune_seed0.png", dpi=150)

# 4. Per-class accuracy, control vs fine-tune, averaged over seeds
acc = {c: [] for c in ["control", "finetune"]}
for s in SEEDS:
    y = np.load(R / f"seed{s}/test_labels.npy")
    for c in acc:
        p = np.load(R / f"seed{s}/{c}_test_probs.npy").argmax(1)
        acc[c].append([(p[y == k] == k).mean() for k in range(10)])
ctrl, ft = np.mean(acc["control"], 0), np.mean(acc["finetune"], 0)
print(f"\n{'class':15s} control  finetune  gain")
for k in np.argsort(ft):
    print(f"{names[k]:15s} {ctrl[k]:.3f}    {ft[k]:.3f}    {ft[k]-ctrl[k]:+.3f}")

# 5. Top 5 confusions (fine-tuned, summed over seeds)
tot = sum(confusion_matrix(np.load(R / f"seed{s}/test_labels.npy"),
                           np.load(R / f"seed{s}/finetune_test_probs.npy").argmax(1)) for s in SEEDS)
np.fill_diagonal(tot, 0)
print("\ntop confusions (true -> predicted, count over 3 seeds of 250 imgs/class):")
for idx in np.argsort(tot, axis=None)[::-1][:5]:
    i, j = np.unravel_index(idx, tot.shape)
    print(f"  {names[i]:13s} -> {names[j]:13s} {tot[i, j]}")