# Scaling Molecular Target Prediction with Machine Learning  
### A Comparative Study of Traditional Models and the MolE Transformer

**Author**: Liujing Chen  
**Affiliation**: Uppsala University, Department of Pharmaceutical Biosciences  
**Thesis Date**: June 2025

---

## Project Overview

This project investigates the large-scale prediction of molecule–target binding affinity using machine learning models. Specifically, it compares three traditional algorithms—FAISS-enhanced K-Nearest Neighbors ([FAISS-KNN](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)), online Random Forest (online RF), and XGBoost—with a modern deep learning foundation model, [MolE](https://doi.org/10.1038/s41467-024-53751-y), which is a Transformer-based architecture pretrained on molecular graphs .

The models were evaluated on a sparse molecule–target interaction matrix derived from ChEMBL (774,357 molecules × 6,142 targets). Both the full dataset and a filtered subset (≥ 20 actives per target) were used for analysis under single-task and multi-task learning frameworks.

---

## Key Findings

- **FAISS-KNN** achieved the best overall performance in both regression and classification tasks. Its simplicity, scalability, and local similarity exploitation proved highly effective, especially in sparse settings.
- **XGBoost** offered strong baseline performance and competitive results despite being a simpler, single-task model.
- **Random Forest** models were trained incrementally in an online setting. While scalable, they struggled with sparse data and yielded the weakest results among all models.
- **MolE**, although state-of-the-art in other molecular tasks, showed only moderate performance here. Pretraining on ~1.2 million molecules brought limited gains compared to training from scratch.
- Filtering the dataset significantly improved both masked MSE and AUC, highlighting the importance of data quality and target coverage.

---

## Method Summary

- **Data**: Molecule–target pairs with experimental pIC₅₀ values were extracted from ChEMBL 34. ECFP4 fingerprints (2048 bits) were generated using RDKit.
- **Models**:
  - **FAISS-KNN**: Multi-task KNN using FAISS for fast nearest neighbor search on large-scale fingerprints.
  - **Random Forest**: Multi-task, online-trained RF using chunked datasets and incremental estimators.
  - **XGBoost**: Single-task regressors trained per target with tree-based gradient boosting.
  - **MolE**: Transformer-based model trained on SMILES strings, evaluated both with and without pretraining.

- **Evaluation**: Performance was assessed using both regression (global and masked MSE) and classification metrics (ROC-AUC, precision, recall, F1-score) at different pIC₅₀ thresholds.

---

## Limitations

- Pretrained MolE weights (~842M molecules) were not available; instead, a smaller model pretrained on ~1.2M GuacaMol molecules was used.
- The ChEMBL-derived interaction matrix is not included due to size limitations.
- Models were originally trained on an academic HPC cluster (Berzelius), which is no longer accessible. This repository reconstructs key scripts based on saved local code.

---

## Contact

For questions or academic correspondence:  
**Email**: Liujing_chen1016@outlook.com *(or replace with your preferred contact)*
