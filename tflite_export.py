import time, os, json, numpy as np, tensorflow as tf
from pathlib import Path

R, OUT = Path("results/seed0"), Path("results/tflite")
OUT.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(R / "finetune.keras")
print("layers:", [l.name for l in model.layers])   # layer 1 should be the augmentation Sequential

# inference graph WITHOUT augmentation layers (shares the trained weights)
inp = tf.keras.Input((224, 224, 3)); x = inp
for layer in model.layers[2:]:
    x = layer(x)
infer = tf.keras.Model(inp, x)
infer.export(str(OUT / "saved_model"))

test = tf.keras.utils.image_dataset_from_directory("data/test", image_size=(224, 224), batch_size=1, shuffle=False)
train = tf.keras.utils.image_dataset_from_directory("data/train", image_size=(224, 224), batch_size=1, shuffle=True, seed=0)
X = np.concatenate([x.numpy().astype(np.uint8) for x, _ in test])
Y = np.concatenate([y.numpy() for _, y in test])

def rep():   # calibration data for int8 comes from TRAIN, never test
    for img, _ in train.take(200):
        yield [tf.cast(img, tf.float32)]

def convert(kind):
    c = tf.lite.TFLiteConverter.from_saved_model(str(OUT / "saved_model"))
    if kind in ("dynamic", "int8"):
        c.optimizations = [tf.lite.Optimize.DEFAULT]
    if kind == "int8":
        c.representative_dataset = rep
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    p = OUT / f"model_{kind}.tflite"
    p.write_bytes(c.convert())
    return p

def evaluate(path):
    it = tf.lite.Interpreter(model_path=str(path), num_threads=1)   # 1 thread ~ mobile-like CPU
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    preds, times = [], []
    for k in range(len(X)):
        it.set_tensor(i["index"], X[k:k + 1].astype(np.float32))
        t = time.perf_counter(); it.invoke(); times.append(time.perf_counter() - t)
        preds.append(it.get_tensor(o["index"]).argmax())
    return float((np.array(preds) == Y).mean()), float(np.median(times) * 1000)

keras_acc = float((infer.predict(X.astype(np.float32), batch_size=64, verbose=0).argmax(1) == Y).mean())
rows = [{"variant": "keras (.keras file)", "size_MB": os.path.getsize(R / "finetune.keras") / 1e6,
         "test_acc": keras_acc, "latency_ms": None}]
for kind in ["float32", "dynamic", "int8"]:
    p = convert(kind)
    acc, ms = evaluate(p)
    rows.append({"variant": f"tflite {kind}", "size_MB": p.stat().st_size / 1e6, "test_acc": acc, "latency_ms": ms})
    print(rows[-1])

json.dump(rows, open(OUT / "tflite_results.json", "w"), indent=2)
print(f"\n{'variant':22s} {'size MB':>8s} {'test acc':>9s} {'ms/img':>8s}")
for r in rows:
    lat = f"{r['latency_ms']:.1f}" if r["latency_ms"] else "-"
    print(f"{r['variant']:22s} {r['size_MB']:8.2f} {r['test_acc']:9.4f} {lat:>8s}")