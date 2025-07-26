# 🏆 NeurIPS 2025 - Open Polymer Prediction Challenge

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)


## 🚀 State-of-the-Art Multi-Task Polymer Property Prediction

This repository contains our cutting-edge solution for the **NeurIPS 2025 Open Polymer Prediction Challenge**. Our approach leverages the latest advances in graph neural networks, transformer architectures, and multi-modal learning to predict five critical polymer properties from SMILES representations.

## 📊 Challenge Overview

**Objective**: Predict 5 polymer properties from chemical structures (SMILES)
- **Tg** (Glass Transition Temperature)
- **FFV** (Fractional Free Volume)
- **Tc** (Crystallization Temperature)
- **Density**
- **Rg** (Radius of Gyration)

**Evaluation Metric**: Weighted Mean Absolute Error (MAE) with property-specific weights

## 🌟 Key Features

### 🧠 Advanced Neural Architecture
- **Graph Transformer Networks** with multi-head attention mechanisms
- **Heterogeneous GNN Ensemble**: GAT, GIN, SAGE, GCN, ChebNet
- **Pre-trained ChemBERTa** transformer for SMILES encoding
- **Uncertainty Quantification** with aleatoric uncertainty estimation

### 🔬 Comprehensive Feature Engineering
- **1600+ Mordred Molecular Descriptors**
- **3D Conformer-based Features** (when applicable)
- **Multiple Molecular Fingerprints**:
  - Morgan (Extended Connectivity)
  - MACCS Keys
  - RDKit Fingerprint
  - Atom Pairs
  - Topological Torsion
- **Graph-level Features**: Wiener Index, Balaban J, Graph Energy

### 🤖 Machine Learning Ensemble
- **XGBoost** with custom objectives
- **LightGBM** with categorical features
- **CatBoost** with GPU acceleration (when available)
- **Random Forest** & **Extra Trees** for stability
- **Stacking Meta-Learner** for optimal combination

### 💡 Intelligent Data Handling
- **Smart Imputation Module**: Chemical similarity-based imputation
- **Structure-Property Relationships**: Physics-informed predictions
- **Masked Loss Functions**: Proper handling of 93.6% missing Tg values
- **Weighted MAE**: Adaptive weighting based on data availability

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neurips-polymer-prediction-2025.git
cd neurips-polymer-prediction-2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
neurips-polymer-prediction-2025/
├── polymer_prediction_sota_final.py   # Main training and prediction script
├── smart_imputation.py                # Intelligent imputation strategies
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── data/
│   ├── dataset1.csv                  # Training data (874 samples)
│   ├── dataset2.csv                  # Training data (7208 samples)
│   ├── dataset3.csv                  # Training data (46 samples)
│   ├── dataset4.csv                  # Test data (862 samples)
│   └── submission_sample.csv         # Submission format
├── cache/                            # Cached features and models
├── checkpoints/                      # Model checkpoints
└── submissions/                      # Generated predictions
```

## 🚀 Usage

### Training and Prediction

```bash
# Run the complete pipeline
python polymer_prediction_sota_final.py

# The script will:
# 1. Load and preprocess data
# 2. Extract molecular features (cached for efficiency)
# 3. Train multi-architecture GNN ensemble
# 4. Train ML ensemble models
# 5. Generate predictions with uncertainty estimates
# 6. Apply intelligent imputation for missing predictions
# 7. Create submission file
```

### Configuration Options

Edit the `Config` class in `polymer_prediction_sota_final.py`:

```python
class Config:
    # Model architecture
    gnn_hidden_dim = 512
    gnn_layers = 6
    num_heads = 8
    dropout = 0.2
    
    # Training
    batch_size = 32
    epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-5
    
    # K-fold cross-validation
    n_folds = 5
    
    # Feature engineering
    use_3d_features = True
    use_mordred = True
    use_transformers = True
```

## 🔬 Technical Approach

### 1. Multi-Modal Feature Extraction
- **Molecular Graphs**: Node features from atomic properties, edge features from bond types
- **SMILES Sequences**: Tokenized and embedded using pre-trained ChemBERTa
- **Descriptors**: Comprehensive set of physicochemical properties
- **Fingerprints**: Multiple representations for similarity computation

### 2. Graph Neural Network Architecture
```
Input SMILES → Molecular Graph → Multi-Architecture GNN Layers → 
→ Graph Pooling → Feature Fusion → Property-Specific Heads → Predictions
```

### 3. Ensemble Strategy
- **Level 1**: Multiple GNN architectures with different inductive biases
- **Level 2**: Gradient boosting ensemble on molecular features
- **Level 3**: Weighted combination based on per-property performance

### 4. Missing Data Strategy
- **Training**: Masked loss functions ignore missing values
- **Inference**: Smart imputation using:
  - Chemical similarity (Tanimoto distance)
  - Structure-property relationships
  - Multi-variate imputation with property correlations

## 📈 Performance

| Property | Data Coverage | OOF MAE | Test MAE |
|----------|--------------|---------|----------|
| Tg       | 3.3%         | TBD     | TBD      |
| FFV      | 46.5%        | TBD     | TBD      |
| Tc       | 4.3%         | TBD     | TBD      |
| Density  | 3.6%         | TBD     | TBD      |
| Rg       | 3.6%         | TBD     | TBD      |

## 🧪 Reproducibility

- **Random Seeds**: Fixed for PyTorch, NumPy, and random modules
- **Feature Caching**: Deterministic feature extraction
- **Checkpointing**: Best models saved per fold
- **Environment**: Tested on Python 3.8-3.11, PyTorch 2.0+

## 📚 References

1. **MMPolymer**: Multi-modal representation learning for polymer property prediction (2024)
2. **Graph Transformers**: Generalization of transformers to graph-structured data
3. **ChemBERTa**: Pre-trained transformer models for molecular property prediction
4. **Mordred**: A molecular descriptor calculator

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏆 Competition Details

- **Competition**: NeurIPS 2025 - Open Polymer Prediction Challenge
- **Prize Pool**: $1,000,000
- **Evaluation**: Weighted MAE across 5 polymer properties
- **Timeline**: TBD

## 💻 Hardware Requirements

- **Minimum**: 16GB RAM, 4-core CPU
- **Recommended**: 32GB RAM, 8-core CPU, GPU (optional but speeds up CatBoost)
- **Storage**: ~10GB for features and model checkpoints

## 🐛 Known Issues

- 3D conformer generation may fail for some complex polymers
- Memory usage can be high with full Mordred descriptor set
- First run will be slow due to feature extraction

## 📞 Contact

For questions or collaborations:
- Create an issue in this repository
- Email: lorenzo.detomasi@outlook.com

---

**Note**: This solution represents state-of-the-art techniques as of 2025, incorporating the latest research in graph neural networks, molecular representation learning, and multi-task learning for polymer property prediction.
