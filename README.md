# Pneumonia Detection with Chest X-ray:

This repository showcases a deep learning–powered medical imaging application that detects **Pneumonia vs Normal** from chest X-ray images using a custom-designed CNN model with **accuracy ~97%**. The model is visualized using Grad-CAM and deployed using **Gradio on Hugging Face Spaces**.

---
## Live Demo

🔗 **[Try the Web App Here](https://thiyaga158-pneumonia-detector.hf.space/)** 

### 📌 Sample: Normal Chest X-ray
![normal](https://github.com/user-attachments/assets/24f96be3-38b7-4dfc-b7e9-74d325d1bd57)

### 📌 Sample: Pneumonia Chest X-ray
![pneumonia](https://github.com/user-attachments/assets/f57d152f-bdd1-4086-8434-e407529f53b0)

---

## Problem Statement

Pneumonia is a serious lung infection that can be fatal if not detected early. Chest X-rays provide a non-invasive and accessible way to diagnose it. The goal of this project is to automatically classify X-rays as **Pneumonia** or **Normal**, and to **visually explain** the model's decision using **Grad-CAM heatmaps**.

---

## 📈 Project Evolution (What I Learned & Improved)

This project went through several thoughtful stages, each driven by experimentation and problem-solving:

### 🧪 Stage 1: Simple CNN (Baseline)
- Started with a simple 3-layer CNN.
- Achieved moderate accuracy (~65%), indicating **underfitting** and lack of model capacity.

### Stage 2: Deep Custom CNN (8 Layers)
- Increased convolutional depth and added **SE Blocks** to better extract features.
- Accuracy improved to ~75%, but still **below expectations**.

### Stage 3: Transfer Learning
- Applied **EfficientNetB0** and **ResNet120** (fine-tuned).
- Showed improved AUC and faster convergence, but models were **less stable** on real-world X-rays.

### Stage 4: Custom DeepResNet Replica
- Built a deeper, ResNet-inspired model with residual connections.
- This gave us more control and interpretability, with performance comparable to pretrained models.

### Realization: Dataset Bottlenecks
- Initial dataset had only **5,216 X-rays**, which led to **overfitting**.
- **Imbalance in Data entries:** ~3.5k Pneumonia vs ~1.5k Normal images.
- Tried to mitigate with:
  - **MixUp augmentation** (smoothing class boundaries)
  - **Focal Loss** (focus on hard samples)
- Accuracy peaked around **88%**, but generalization issues remained.
- Also noticed **padding color** during resizing affected performance.

### Stage 5: Dataset Augmentation & Expansion
- Combined and cleaned multiple pneumonia datasets.
- Final dataset: **22,000+ high-quality, balanced images**.
- Initially used images of size **512×512**, but later resized to **224×224** to reduce computational load as i increased my dataset to a large extent.
- All images resized to 224×224 using a custom **resize + padding script**.
- After experiments, switched to **black padding** for consistency.
- Switched to **CBAM attention blocks** instead of **SE blocks** in ImprovedPneumonia Model(Custom CNN model).

---

## 🖼️ Padding Strategy Evaluation

We experimented with various padding colors during image resizing to 224×224. Results:

| Padding Color | Accuracy | Observations |
|---------------|----------|--------------|
| **White**   | ~88%     | Too bright; disrupted chest region contrast. |
| **Gray**    | ~90%     | Better contrast but introduced visual noise. |
| **Black**   | **~97%** | Best visual integrity; minimized distractions. |

**Black padding was adopted** as the final strategy for both training and deployment.

---

### Final Model (This Repo)
- Enhanced ImprovedPneumonia with:
  - **CBAM attention modules**
  - **Dropout regularization**
  - **SiLU activations**
  - **Grad-CAM integration**
The ensemble model (built with these custom models and EfficientNetB0):
- Achieved:
  - AUC: **0.9973**
  - F1 Score: **0.9724**
  - Accuracy: **97.18%**

---

## Grad-CAM Integration

We used **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize the model’s decision-making process by highlighting important regions in the input image.

### Preprocessing:
We included:
- Image resizing with **black padding** to 224x224
- Grayscale conversion
- Normalization (mean=0.5, std=0.5)

---

## Deployment (Gradio + Hugging Face Spaces)

We deployed this using **Gradio** and hosted it on **Hugging Face Spaces**.

### Steps:
1. Created a `gradio-deploy/` folder with:
   - `app.py`
   - `requirements.txt`
   - `ImprovedPneumoniaCNN.pth`

2. Included:
   - A resize and padding function to ensure deployment images match training preprocessing.

3. Uploaded everything to [Hugging Face Spaces](https://huggingface.co/spaces) using the **Blank template**.

4. Fixed CORS/localhost issues with:
```python
interface.launch(share=True)
```
---

## License

This project is licensed under the [Apache 2.0 License](LICENSE)..

---

