## 🌐 MolE Model: Integration & Fine-tuning

### 📌 What is MolE?

[MolE](https://doi.org/10.1038/s41467-024-53751-y) is a Transformer-based foundation model for molecular property prediction. It is pretrained on millions of molecular graphs using disentangled attention, and achieves state-of-the-art performance across many ADMET tasks.

The original model and weights were made publicly available by the authors and are not developed in this project.

---

### 🧪 My Contribution

Although MolE is a public foundation model, this project focuses on its **adaptation, fine-tuning, and comparative evaluation** for a novel task setting: **large-scale multi-target binding affinity prediction**.

**My contributions include:**

- ✅ Independently setting up the **fine-tuning pipeline** for MolE on my own molecular-target interaction matrix (from ChEMBL)
- ✅ Adapting input preprocessing to match MolE's expected SMILES format and tokenizer
- ✅ Testing and comparing **two different MolE configurations**:
  - Pretrained on ~1.2M GuacaMol molecules
  - Trained from scratch on my dataset
- ✅ Designing **multi-metric evaluation** (MSE, AUC, etc.) to benchmark against traditional models (KNN, XGBoost, RF)
- ✅ Analyzing the **effect of pretraining** in the context of sparse regression targets
- ✅ Architecture-Level Exploration and Module Customization: 
In addition to applying the MolE model as released, I experimented with modifying several components of the architecture to better adapt it to the large-scale binding affinity prediction task. These explorations included:

- **Alternative loss functions**: such as using Huber Loss instead of MSE to reduce sensitivity to outliers in sparse target labels.
- **Prediction head modifications**: including adding a multi-layer perceptron (MLP) layer before output, or comparing shared vs. separate heads across tasks.
- **Positional encoding schemes**: testing whether disabling or modifying positional encodings impacts model generalization on unordered SMILES strings.
- **Attention mechanism variants**: adjusting the number of attention heads or exploring sparse vs. dense attention patterns to improve scalability on long inputs.
- **Dropout and regularization**: tuning dropout rates and layer normalization configurations for better generalization in low-data settings.

Although not all configurations were retained as finalized scripts, this process reflects my ability to go beyond plug-and-play usage of foundation models. It demonstrates a solid understanding of model internals, the initiative to explore architectural tuning, and the capacity to connect model behavior with task-specific constraints.

---




### 🔄 Model Setup in This Project

- **Model weights used**: `mole-1.2M-guacamol.pt` (publicly released)
- **Training strategy**: Multi-task regression with MSE loss
- **Preprocessing**: Tokenized SMILES using the official MolE tokenizer
- **Training environment**: Remote HPC cluster (Berzelius) using PyTorch 2.2

---

### ⚠️ Note

This repository **does not claim authorship** of the MolE architecture or pretraining methodology.  
All credit for model development goes to the original authors.  
This project demonstrates how such foundation models can be critically integrated, benchmarked, and analyzed in a real-world predictive context.

---

### 🔗 Reference

Méndez-Lucio, O., Nicolaou, C. A., & Earnshaw, B. (2024). [MolE: A foundation model for molecular graphs using disentangled attention](https://doi.org/10.1038/s414)
