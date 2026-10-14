# "Tell me about this project": Interview Answer

## 60-second version (say this first)

"I built an image classifier for 10 food categories from the Food-101 dataset. I used transfer learning on MobileNetV2 in TensorFlow/Keras and then deployed it as an int8 TensorFlow Lite model.

I didn't want to just report an accuracy number, so I framed the project around three questions. First, does fine-tuning actually beat a frozen base if both get the same training budget? Second, where does the model fail, and why? Third, what does int8 quantization cost for on-device deployment?

The answers: fine-tuning reached **81.0%** test accuracy, versus **79.0%** for a frozen-base control trained for the same number of epochs. The gain held on all 3 seeds. About 70% of that gain came from a single class, and when I dug in, it was mostly a shifted decision boundary rather than better features. Full int8 quantization made the model **3.3× smaller (9.5 → 2.9 MB) and 1.5× faster (3.9 → 2.6 ms per image)** with no measurable loss in accuracy."

Then stop and let them choose what to dig into.

---

## Deeper version (2–3 minutes, if they want more)

**1. Setup and data**
- 10 classes from Food-101, using the official train/test split. I deliberately included 4 visually similar meats (steak, filet mignon, pork chop, prime rib) so the confusion matrix would have something to show, plus 6 easier dishes.
- 6,000 train, 1,500 validation and 2,500 test images. The test set was never used for tuning or model selection.
- Model: MobileNetV2 (ImageNet weights) → GlobalAveragePooling → Dense(128) → Dropout(0.3) → softmax. Augmentation used flip, rotation and zoom.

**2. Experimental design: the part I'm proudest of**
- Stage 1: frozen base, 10 epochs, learning rate 1e-3.
- **Control:** kept the base frozen for 10 *more* epochs.
- **Fine-tune:** started from the *same* Stage 1 weights, unfroze the last 30 layers, learning rate 1e-5, 10 epochs.
- Each condition ran on 3 seeds with an identical data split. Many tutorials compare "fine-tuned for 20 epochs" against "frozen for 10 epochs", which mixes up fine-tuning with simply training longer. With my control, the extra frozen epochs added only +0.2 points, so the +1.9 gain really comes from adapting the base network. The ranges don't overlap: the worst fine-tuned run (80.6%) still beat the best control run (79.4%).

**3. A real bug I found and fixed**
- In the first fine-tuning epoch, training accuracy dropped from 0.81 to 0.65, even though validation accuracy showed the weights had loaded correctly.
- The cause was the unfrozen BatchNorm layers switching to batch statistics. I froze all BatchNorm layers explicitly, and epoch 1 then started at about 0.82 on every seed.
- I also made sure inputs were scaled to [-1, 1], which is what MobileNetV2 expects. Getting this wrong is a common silent bug.

**4. Error analysis: being honest about the gain**
- All of the top 5 confusions are between the meat classes. Steak is the worst class at 32% accuracy.
- Filet mignon recall rose from 52% to 65%, but its precision stayed flat at about 52–53%. The model simply predicted "filet mignon" 22% more often, mostly at steak's expense. So the gain is real and consistent, but it's a boundary shift, not broadly better features.
- Root cause: filet mignon *is* a cut of steak, and Food-101's training labels are deliberately left uncleaned. Many of these "errors" are images where either label is defensible, so this is a label-definition problem more than a model problem.

**5. Deployment**
- I exported the model without its augmentation layers and converted it to three TFLite variants: float32, dynamic-range and full int8. Full int8 was calibrated on 200 *training* images, never test images.
- TFLite float32 matched the Keras model to within 2 images, so the conversion was faithful.
- Full int8: 2.9 MB, 81.6% accuracy, 2.6 ms per image. Dynamic-range was small but *slower* (4.7 ms), because it converts weights back to float at runtime. Only full int8 actually speeds up inference.

**6. Limitations (say these before they ask)**
- I didn't tune the hyperparameters, so the fine-tuning gain is a lower bound.
- Latency was measured on a desktop CPU (1 thread, XNNPACK), not on a phone. The relative int8 speedup is the part that transfers.
- This is a 10-class subset, so the results don't carry over directly to the full 101-class task.

---

## Likely follow-up questions

- **Why MobileNetV2?** It's built for mobile and edge devices (depthwise separable convolutions), it's small, and it quantizes well. That fit the deployment goal.
- **Why such a low learning rate for fine-tuning (1e-5)?** A large learning rate would wreck the pretrained features, which is catastrophic forgetting. Small updates adapt them gently.
- **Why is validation accuracy (~79%) lower than test accuracy (~81%)?** Validation was carved from Food-101's noisy training labels, while the test set was manually verified.
- **Why was the .keras file 23.7 MB?** It stores the optimizer state too. The fair baseline is TFLite float32 at 9.55 MB.
- **What would you do next?** Tune the unfreeze depth and learning rate, try label smoothing or cleaning the steak and filet-mignon labels, measure latency on a real phone, and scale up to all 101 classes.
- **How did you avoid leaking test data?** Test images were used only for final scoring. The validation split used a fixed seed, and int8 calibration used training images.

## Tech stack
Python, TensorFlow/Keras, MobileNetV2, TFLite (int8 quantization), NumPy, scikit-learn, Matplotlib. Trained on an RTX 3050 (about 8 minutes per seed).
