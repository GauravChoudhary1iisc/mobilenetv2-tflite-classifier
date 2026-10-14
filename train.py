import sys, json, numpy as np, tensorflow as tf
from pathlib import Path

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
E1, E2, N_UNFREEZE = 10, 10, 30
IMG, BS = (224, 224), 32
tf.keras.utils.set_random_seed(SEED)
out = Path(f"results/seed{SEED}"); out.mkdir(parents=True, exist_ok=True)

# split seed fixed at 42 -> identical train/val split across all runs
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train", validation_split=0.2, subset="both", seed=42, image_size=IMG, batch_size=BS)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test", image_size=IMG, batch_size=BS, shuffle=False)
class_names = train_ds.class_names
AUTO = tf.data.AUTOTUNE
train_ds, val_ds, test_ds = [d.prefetch(AUTO) for d in (train_ds, val_ds, test_ds)]

def build():
    base = tf.keras.applications.MobileNetV2(input_shape=IMG + (3,), include_top=False, weights="imagenet")
    base.trainable = False
    aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),
                               tf.keras.layers.RandomRotation(0.1),
                               tf.keras.layers.RandomZoom(0.1)])
    inp = tf.keras.Input(IMG + (3,))
    x = aug(inp)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)   # = mobilenet_v2.preprocess_input, [-1, 1]
    x = base(x, training=False)                               # BatchNorm stays in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outp = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    return tf.keras.Model(inp, outp), base

def fit(model, lr, epochs, name):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
    json.dump(h.history, open(out / f"{name}_history.json", "w"))

def test_eval(model, name):
    probs = model.predict(test_ds, verbose=0)
    y = np.concatenate([y for _, y in test_ds])
    np.save(out / f"{name}_test_probs.npy", probs)
    np.save(out / "test_labels.npy", y)
    return float((probs.argmax(1) == y).mean())

res = {}
# Stage 1: frozen base
model, base = build()
fit(model, 1e-3, E1, "stage1")
model.save_weights(out / "stage1.weights.h5")
res["stage1_frozen_10ep"] = test_eval(model, "stage1")

# Control: stay frozen, train E2 more epochs (same total epochs as fine-tune)
fit(model, 1e-3, E2, "control")
res["control_frozen_20ep"] = test_eval(model, "control")

# Fine-tune: from the SAME stage-1 weights, unfreeze last N layers, low LR
model, base = build()
model.load_weights(out / "stage1.weights.h5")
base.trainable = True
for layer in base.layers[:-N_UNFREEZE]:
    layer.trainable = False
for layer in base.layers:                                          # NEW: keep ALL BatchNorm frozen
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])   # NEW: needed for evaluate
print("check - stage1 weights loaded, val [loss, acc]:",                      # NEW: should be ~0.757
      model.evaluate(val_ds, verbose=0))
fit(model, 1e-5, E2, "finetune")
res["finetune_20ep"] = test_eval(model, "finetune")
model.save(out / "finetune.keras")

json.dump({**res, "class_names": class_names}, open(out / "summary.json", "w"), indent=2)
print(res)