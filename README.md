# 🧠 Simple PyTorch Autoencoder

A lightweight, beginner-friendly PyTorch project implementing an **Autoencoder** from scratch. 

The primary goal of this project is to explore how neural networks compress data through a bottleneck layer, reconstruct it so that predictions match the inputs, and leverage **backpropagation** using PyTorch.

---

## 🎯 Project Purpose & Intuition

Have you ever wondered how large language models (LLMs) or deep neural networks store and remember information? 
* **Data Compression & Latent Space:** This project serves as a foundational step. By forcing high-dimensional data through a strict "bottleneck" (a smaller hidden layer), the network is forced to learn the most essential patterns and discard noise.
* **Self-Supervised Learning:** Since the inputs and predictions are identical ($X = \hat{X}$), the network learns without explicit external labels, purely by minimizing its reconstruction error.

---

## 🔬 Key Findings & Empirical Insights

### Impact of Network Depth on Reconstruction Loss

During experimentation, we observed a significant relationship between the network's **llm_depth** and its ability to minimize the loss:

1. **Shallow Architecture (Low Depth):**
   * **Observation:** When using fewer/shallower hidden layers, the model's loss plateaus quickly and fails to decrease past a certain threshold.
   * **Why:** A shallow autoencoder lacks the non-linear capacity to model complex relationships within the data through a compressed bottleneck. It suffers from **underfitting** because the expressive power of the network is insufficient.

2. **Deeper Architecture (High Depth):**
   * **Observation:** Increasing the network depth consistently lowered the reconstruction loss to zero levels.
   * **Why:** Deeper networks create a richer hierarchical representation. Each additional layer allows the model to perform incremental non-linear transformations (`LeakyReLU`), making it easier to project high-dimensional data into a low-dimensional latent space and accurately reconstruct it back.

> **Conclusion:** Just as Large Language Models (LLMs) rely on deep transformer stacks to capture high-level semantic abstractions, autoencoders require sufficient depth to effectively compress and reconstruct data without losing critical information.

---

## 🏗️ Model Architecture

The architecture consists of two main parts connected by a low-dimensional bottleneck:

1. **Encoder:** Compresses the input features down to a lower-dimensional latent space.
   * `Linear(d_model -> llm_depth)` $\rightarrow$ `LeakyReLU`
2. **Decoder:** Reconstructs the compressed latent representation back into the original feature dimensions.
   * `Linear(llm_depth -> d_model)` $\rightarrow$ `LeakyReLU`
