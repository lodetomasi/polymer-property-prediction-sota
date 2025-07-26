#!/usr/bin/env python3
"""
NeurIPS 2025 - Open Polymer Prediction Challenge
COMPLETE STATE-OF-THE-ART IMPLEMENTATION
Based on latest research (2024-2025):
- MMPolymer-style multimodal approach
- Advanced Graph Neural Networks
- Weighted MAE for multi-task learning
- Handles missing values correctly
"""

import os
import gc
import time
import warnings
import random
import pickle
import json
import hashlib
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import lru_cache
from collections import defaultdict

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# PyTorch Geometric
try:
    from torch_geometric.nn import GATConv, GINConv, SAGEConv, GCNConv, ChebConv, GraphConv
    from torch_geometric.nn import TransformerConv, GPSConv, GatedGraphConv
    from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, global_sort_pool
    from torch_geometric.nn import BatchNorm, LayerNorm, GraphNorm
    from torch_geometric.data import Data, Batch
except ImportError:
    print("Installing PyTorch Geometric...")
    os.system("pip install torch-geometric")
    from torch_geometric.nn import GATConv, GINConv, SAGEConv, GCNConv, ChebConv, GraphConv
    from torch_geometric.nn import TransformerConv, GPSConv, GatedGraphConv
    from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, global_sort_pool
    from torch_geometric.nn import BatchNorm, LayerNorm, GraphNorm
    from torch_geometric.data import Data, Batch

# RDKit
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, Fragments, Lipinski, Crippen
    from rdkit.Chem import rdMolTransforms, rdPartialCharges, rdmolops
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("Installing RDKit...")
    os.system("pip install rdkit")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, Fragments, Lipinski, Crippen
    from rdkit.Chem import rdMolTransforms, rdPartialCharges, rdmolops
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

# Mordred
try:
    from mordred import Calculator, descriptors as mordred_descriptors
except ImportError:
    print("Installing Mordred...")
    os.system("pip install mordred")
    from mordred import Calculator, descriptors as mordred_descriptors

# ML libraries
try:
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
except ImportError:
    print("Installing ML libraries...")
    os.system("pip install lightgbm xgboost catboost")
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

# Sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

# Transformers
try:
    from transformers import AutoTokenizer, AutoModel, RobertaModel, RobertaTokenizer
except ImportError:
    print("Installing transformers...")
    os.system("pip install transformers")
    from transformers import AutoTokenizer, AutoModel, RobertaModel, RobertaTokenizer

# Import smart imputation module
from smart_imputation import create_intelligent_predictions

# Configuration
class Config:
    # Device
    device = torch.device('cpu')
    
    # Paths
    train_path = 'train.csv'
    test_path = 'test.csv'
    supplement_path = 'train_supplement'
    submission_path = 'submission.csv'
    checkpoint_dir = 'checkpoints'
    cache_dir = 'cache'
    
    # Model parameters
    hidden_dim = 512
    gnn_hidden_dim = 256
    n_gnn_layers = 6
    n_attention_heads = 8
    dropout = 0.2
    batch_size = 16
    learning_rate = 1e-3
    epochs = 100
    patience = 15
    gradient_accumulation_steps = 4
    
    # Training
    n_folds = 5
    seed = 42
    n_jobs = max(1, cpu_count() - 2)
    use_cache = True
    
    # Properties to predict
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    # Weighted MAE based on data availability
    property_weights = {
        'Tg': 1.0,      # 6.4% data
        'FFV': 2.0,     # 88.2% data - most important
        'Tc': 1.0,      # 9.2% data
        'Density': 1.5, # 7.7% data
        'Rg': 1.0       # 7.7% data
    }
    
    # Feature extraction
    use_mordred = True  # Full Mordred descriptors
    use_3d = True       # 3D conformer features
    use_transformer = True  # ChemBERTa/RoBERTa
    max_seq_length = 512
    
    # Advanced features
    use_graph_transformer = True
    use_attention_pooling = True
    use_uncertainty = True
    
    # Physical constraints for polymers
    constraints = {
        'Tg': (-150, 500),      # Glass transition temperature (°C)
        'FFV': (0.0, 1.0),      # Fractional free volume
        'Tc': (0.0, 1.0),       # Crystallization temperature (normalized)
        'Density': (0.5, 3.0),  # Density (g/cm³)
        'Rg': (5.0, 50.0)       # Radius of gyration (nm)
    }

config = Config()

# Create directories
os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.cache_dir, exist_ok=True)

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(config.seed)

# ============================================================================
# ADVANCED FEATURE EXTRACTION
# ============================================================================

class AdvancedMolecularFeatureExtractor:
    """Extract comprehensive molecular features including Mordred and 3D"""
    
    def __init__(self, use_mordred=True, use_3d=True, cache_dir='cache'):
        self.use_mordred = use_mordred
        self.use_3d = use_3d
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        if use_mordred:
            self.mordred_calc = Calculator(mordred_descriptors, ignore_3D=not use_3d)
    
    def extract_features(self, smiles):
        """Extract all features with caching"""
        # Cache key based on SMILES and settings
        cache_key = hashlib.md5(f"{smiles}_{self.use_mordred}_{self.use_3d}".encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}_advanced.npy")
        
        if os.path.exists(cache_path) and config.use_cache:
            return np.load(cache_path)
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Failed to parse SMILES: {smiles}")
                return np.zeros(5000)  # Return zeros for failed molecules
            
            features = []
            
            # 1. Basic RDKit descriptors
            basic_features = self._extract_basic_features(mol)
            features.extend(basic_features)
            
            # 2. Fingerprints
            fp_features = self._extract_fingerprints(mol)
            features.extend(fp_features)
            
            # 3. Fragment features
            fragment_features = self._extract_fragment_features(mol)
            features.extend(fragment_features)
            
            # 4. Polymer-specific features
            polymer_features = self._extract_polymer_features(mol)
            features.extend(polymer_features)
            
            # 5. Mordred descriptors
            if self.use_mordred:
                mordred_features = self._extract_mordred_features(mol)
                features.extend(mordred_features)
            
            # 6. 3D features
            if self.use_3d:
                features_3d = self._extract_3d_features(mol)
                features.extend(features_3d)
            
            # Convert to numpy array
            features = np.array(features, dtype=np.float32)
            
            # Handle NaN and Inf
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Truncate or pad
            if len(features) > 10000:
                features = features[:10000]
            else:
                features = np.pad(features, (0, 10000 - len(features)), constant_values=0)
            
            # Save to cache
            np.save(cache_path, features)
            return features
            
        except Exception as e:
            print(f"Error extracting features for {smiles}: {e}")
            return np.zeros(10000)
    
    def _extract_basic_features(self, mol):
        """Extract basic molecular descriptors"""
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.NumAromaticHeterocycles(mol),
            Descriptors.NumSaturatedHeterocycles(mol),
            Descriptors.NumAliphaticHeterocycles(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumSaturatedCarbocycles(mol),
            Descriptors.NumAliphaticCarbocycles(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0(mol),
            Descriptors.Chi0n(mol),
            Descriptors.Chi0v(mol),
            Descriptors.Chi1(mol),
            Descriptors.Chi1n(mol),
            Descriptors.Chi1v(mol),
            Descriptors.Chi2n(mol),
            Descriptors.Chi2v(mol),
            Descriptors.Chi3n(mol),
            Descriptors.Chi3v(mol),
            Descriptors.Chi4n(mol),
            Descriptors.Chi4v(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
            Descriptors.LabuteASA(mol),
            Descriptors.PEOE_VSA1(mol),
            Descriptors.PEOE_VSA2(mol),
            Descriptors.PEOE_VSA3(mol),
            Descriptors.PEOE_VSA4(mol),
            Descriptors.PEOE_VSA5(mol),
            Descriptors.PEOE_VSA6(mol),
            Descriptors.PEOE_VSA7(mol),
            Descriptors.PEOE_VSA8(mol),
            Descriptors.PEOE_VSA9(mol),
            Descriptors.PEOE_VSA10(mol),
            Descriptors.PEOE_VSA11(mol),
            Descriptors.PEOE_VSA12(mol),
            Descriptors.PEOE_VSA13(mol),
            Descriptors.PEOE_VSA14(mol),
            Descriptors.SMR_VSA1(mol),
            Descriptors.SMR_VSA2(mol),
            Descriptors.SMR_VSA3(mol),
            Descriptors.SMR_VSA4(mol),
            Descriptors.SMR_VSA5(mol),
            Descriptors.SMR_VSA6(mol),
            Descriptors.SMR_VSA7(mol),
            Descriptors.SMR_VSA8(mol),
            Descriptors.SMR_VSA9(mol),
            Descriptors.SMR_VSA10(mol),
            Descriptors.SlogP_VSA1(mol),
            Descriptors.SlogP_VSA2(mol),
            Descriptors.SlogP_VSA3(mol),
            Descriptors.SlogP_VSA4(mol),
            Descriptors.SlogP_VSA5(mol),
            Descriptors.SlogP_VSA6(mol),
            Descriptors.SlogP_VSA7(mol),
            Descriptors.SlogP_VSA8(mol),
            Descriptors.SlogP_VSA9(mol),
            Descriptors.SlogP_VSA10(mol),
            Descriptors.SlogP_VSA11(mol),
            Descriptors.SlogP_VSA12(mol),
            Descriptors.EState_VSA1(mol),
            Descriptors.EState_VSA2(mol),
            Descriptors.EState_VSA3(mol),
            Descriptors.EState_VSA4(mol),
            Descriptors.EState_VSA5(mol),
            Descriptors.EState_VSA6(mol),
            Descriptors.EState_VSA7(mol),
            Descriptors.EState_VSA8(mol),
            Descriptors.EState_VSA9(mol),
            Descriptors.EState_VSA10(mol),
            Descriptors.EState_VSA11(mol),
            Descriptors.VSA_EState1(mol),
            Descriptors.VSA_EState2(mol),
            Descriptors.VSA_EState3(mol),
            Descriptors.VSA_EState4(mol),
            Descriptors.VSA_EState5(mol),
            Descriptors.VSA_EState6(mol),
            Descriptors.VSA_EState7(mol),
            Descriptors.VSA_EState8(mol),
            Descriptors.VSA_EState9(mol),
            Descriptors.VSA_EState10(mol),
            Descriptors.MinAbsEStateIndex(mol),
            Descriptors.MaxAbsEStateIndex(mol),
            Descriptors.MinEStateIndex(mol),
            Descriptors.MaxEStateIndex(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            rdMolDescriptors.CalcNumSpiroAtoms(mol),
            rdMolDescriptors.CalcNumAmideBonds(mol),
            rdMolDescriptors.CalcNumHeterocycles(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
        ]
    
    def _extract_fingerprints(self, mol):
        """Extract various molecular fingerprints"""
        features = []
        
        # Morgan fingerprints with different radii
        for radius in [2, 3, 4]:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024)
            fp_arr = np.zeros(1024)
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
            features.extend(fp_arr)
        
        # MACCS keys
        maccs = AllChem.GetMACCSKeysFingerprint(mol)
        maccs_arr = np.zeros(167)
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
        features.extend(maccs_arr)
        
        # RDKit fingerprint
        rdkit_fp = Chem.RDKFingerprint(mol)
        rdkit_arr = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(rdkit_fp, rdkit_arr)
        features.extend(rdkit_arr)
        
        # Atom pair fingerprint
        ap_fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)
        ap_arr = np.zeros(1024)
        DataStructs.ConvertToNumpyArray(ap_fp, ap_arr)
        features.extend(ap_arr)
        
        # Topological torsion fingerprint
        tt_fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=1024)
        tt_arr = np.zeros(1024)
        DataStructs.ConvertToNumpyArray(tt_fp, tt_arr)
        features.extend(tt_arr)
        
        return features
    
    def _extract_fragment_features(self, mol):
        """Extract fragment-based features"""
        fragment_features = []
        
        # All available fragment descriptors
        for name, func in Fragments.fns:
            try:
                value = func(mol)
                fragment_features.append(value)
            except:
                fragment_features.append(0)
        
        return fragment_features
    
    def _extract_polymer_features(self, mol):
        """Extract polymer-specific features"""
        features = []
        
        # Molecular size and complexity
        features.extend([
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            mol.GetNumHeavyAtoms(),
            len(Chem.GetSymmSSSR(mol)),  # Number of smallest rings
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
        ])
        
        # Polymer-relevant indices
        features.extend([
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
            rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
            rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        ])
        
        # Flexibility indices
        try:
            features.append(rdMolDescriptors.CalcNumRotatableBonds(mol) / mol.GetNumBonds())
        except:
            features.append(0)
        
        # Aromaticity ratio
        try:
            aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            features.append(aromatic_atoms / mol.GetNumAtoms())
        except:
            features.append(0)
        
        # Heteroatom ratio
        try:
            heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
            features.append(heteroatoms / mol.GetNumAtoms())
        except:
            features.append(0)
        
        # Ring complexity
        try:
            ring_info = mol.GetRingInfo()
            features.extend([
                len(ring_info.AtomRings()),
                len(ring_info.BondRings()),
                max([len(ring) for ring in ring_info.AtomRings()]) if ring_info.AtomRings() else 0,
                min([len(ring) for ring in ring_info.AtomRings()]) if ring_info.AtomRings() else 0,
            ])
        except:
            features.extend([0, 0, 0, 0])
        
        return features
    
    def _extract_mordred_features(self, mol):
        """Extract Mordred descriptors"""
        try:
            # Calculate all Mordred descriptors
            result = self.mordred_calc(mol)
            
            # Convert to list, handling errors
            features = []
            for i, desc in enumerate(result):
                try:
                    if isinstance(desc, (int, float)) and not np.isnan(desc) and not np.isinf(desc):
                        features.append(float(desc))
                    else:
                        features.append(0.0)
                except:
                    features.append(0.0)
            
            return features[:1600]  # Limit to 1600 features
        except Exception as e:
            print(f"Mordred extraction error: {e}")
            return [0.0] * 1600
    
    def _extract_3d_features(self, mol):
        """Extract 3D conformer-based features"""
        features = []
        
        try:
            # Generate 3D conformer
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, randomSeed=42, maxAttempts=10)
            
            if mol_3d.GetNumConformers() > 0:
                AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
                
                # 3D descriptors
                features.extend([
                    rdMolDescriptors.CalcPMI1(mol_3d),
                    rdMolDescriptors.CalcPMI2(mol_3d),
                    rdMolDescriptors.CalcPMI3(mol_3d),
                    rdMolDescriptors.CalcNPR1(mol_3d),
                    rdMolDescriptors.CalcNPR2(mol_3d),
                    rdMolDescriptors.CalcRadiusOfGyration(mol_3d),
                    rdMolDescriptors.CalcInertialShapeFactor(mol_3d),
                    rdMolDescriptors.CalcEccentricity(mol_3d),
                    rdMolDescriptors.CalcAsphericity(mol_3d),
                    rdMolDescriptors.CalcSpherocityIndex(mol_3d),
                ])
                
                # Get 3D distance matrix
                conf = mol_3d.GetConformer()
                dist_matrix = AllChem.Get3DDistanceMatrix(mol_3d)
                
                # Statistical features from distance matrix
                features.extend([
                    np.mean(dist_matrix),
                    np.std(dist_matrix),
                    np.max(dist_matrix),
                    np.min(dist_matrix[dist_matrix > 0]),
                    np.percentile(dist_matrix, 25),
                    np.percentile(dist_matrix, 50),
                    np.percentile(dist_matrix, 75),
                ])
            else:
                features.extend([0.0] * 17)
                
        except Exception as e:
            features.extend([0.0] * 17)
        
        return features

def mol_to_advanced_graph(smiles):
    """Convert SMILES to advanced graph representation"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return empty graph
        return Data(x=torch.zeros(1, 150, dtype=torch.float32), 
                   edge_index=torch.zeros(2, 0, dtype=torch.long),
                   edge_attr=torch.zeros(0, 10, dtype=torch.float32))
    
    # Node features (extended)
    atom_features = []
    for atom in mol.GetAtoms():
        features = []
        
        # Basic features
        features.append(atom.GetAtomicNum())
        features.append(atom.GetDegree())
        features.append(atom.GetFormalCharge())
        features.append(int(atom.GetHybridization()))
        features.append(int(atom.GetIsAromatic()))
        features.append(atom.GetTotalNumHs())
        features.append(int(atom.IsInRing()))
        
        # Ring size features
        for ring_size in range(3, 9):
            features.append(int(atom.IsInRingSize(ring_size)))
        
        # Additional features
        features.append(atom.GetNumRadicalElectrons())
        features.append(int(atom.GetChiralTag()))
        features.append(atom.GetMass())
        features.append(Crippen.MolLogP(mol) / mol.GetNumAtoms())  # Atom contribution to LogP
        features.append(Descriptors.TPSA(mol) / mol.GetNumAtoms())  # Atom contribution to TPSA
        
        # One-hot encoding of common atoms (C, N, O, S, P, F, Cl, Br, I)
        atom_type = atom.GetSymbol()
        common_atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        for a in common_atoms:
            features.append(int(atom_type == a))
        
        atom_features.append(features)
    
    # Pad features to fixed size
    atom_features = np.array(atom_features, dtype=np.float32)
    n_features = atom_features.shape[1]
    if n_features < 150:
        padding = np.zeros((atom_features.shape[0], 150 - n_features))
        atom_features = np.hstack([atom_features, padding])
    
    # Edge features
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Bond features
        bond_feats = [
            bond.GetBondTypeAsDouble(),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo()),
        ]
        
        # Add bond features for both directions
        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([bond_feats, bond_feats])
    
    # Convert to tensors
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    x = torch.tensor(atom_features, dtype=torch.float32)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    
    # Pad edge features
    if edge_attr.shape[1] < 10:
        padding = torch.zeros(edge_attr.shape[0], 10 - edge_attr.shape[1])
        edge_attr = torch.cat([edge_attr, padding], dim=1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ============================================================================
# ADVANCED NEURAL NETWORK ARCHITECTURES
# ============================================================================

class AttentionPooling(nn.Module):
    """Attention-based graph pooling"""
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, batch):
        # Compute attention scores
        scores = self.attention(x)
        scores = F.softmax(scores, dim=0)
        
        # Weighted pooling
        pooled = global_add_pool(x * scores, batch)
        return pooled

class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer"""
    
    def __init__(self, input_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = TransformerConv(
            input_dim, 
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=10,
            beta=True
        )
        self.norm = GraphNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr):
        # Multi-head attention
        x_att = self.attention(x, edge_index, edge_attr)
        x = x + self.dropout(x_att)
        x = self.norm(x)
        
        # Feed-forward
        x_ffn = self.ffn(x)
        x = x + self.dropout(x_ffn)
        
        return x

class AdvancedGNN(nn.Module):
    """State-of-the-art Graph Neural Network with multiple architectures"""
    
    def __init__(self, node_dim=150, edge_dim=10, hidden_dim=256, output_dim=256, 
                 n_layers=6, n_heads=8, dropout=0.2):
        super().__init__()
        
        # Initial projection
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        # Multiple GNN architectures
        self.gnn_layers = nn.ModuleList()
        
        for i in range(n_layers):
            if i % 3 == 0:
                # Graph Transformer layers
                layer = GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout)
            elif i % 3 == 1:
                # GAT layers
                layer = GATConv(
                    hidden_dim, 
                    hidden_dim // n_heads,
                    heads=n_heads,
                    dropout=dropout,
                    edge_dim=hidden_dim
                )
            else:
                # GIN layers
                layer = GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.BatchNorm1d(hidden_dim * 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim * 2, hidden_dim)
                    ),
                    train_eps=True
                )
            
            self.gnn_layers.append(layer)
        
        # Normalization layers
        self.norms = nn.ModuleList([
            GraphNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        # Pooling
        self.attention_pool = AttentionPooling(hidden_dim, hidden_dim // 2)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Encode nodes and edges
        x = self.node_encoder(x)
        edge_attr_encoded = self.edge_encoder(edge_attr)
        
        # Store intermediate representations
        representations = []
        
        # Apply GNN layers
        for i, (layer, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            if isinstance(layer, GraphTransformerLayer):
                x_new = layer(x, edge_index, edge_attr_encoded)
            elif isinstance(layer, GATConv):
                x_new = layer(x, edge_index, edge_attr_encoded)
            else:  # GIN
                x_new = layer(x, edge_index)
            
            x = x + self.dropout(x_new)  # Residual connection
            x = norm(x)
            
            if i % 2 == 0:  # Store every other layer
                representations.append(x)
        
        # Multiple pooling strategies
        pool_mean = global_mean_pool(x, batch)
        pool_max = global_max_pool(x, batch)
        pool_add = global_add_pool(x, batch)
        pool_attention = self.attention_pool(x, batch)
        
        # Combine all pooling
        pooled = torch.cat([pool_mean, pool_max, pool_add, pool_attention], dim=1)
        
        # Output projection
        out = self.output_proj(pooled)
        
        return out

class PolymerTransformer(nn.Module):
    """Transformer for SMILES sequences"""
    
    def __init__(self, model_name='seyonec/ChemBERTa-zinc-base-v1', hidden_dim=768, output_dim=256):
        super().__init__()
        
        # Load pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, smiles_list):
        # Tokenize SMILES
        encoded = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors='pt'
        ).to(config.device)
        
        # Get transformer output
        with torch.no_grad() if not self.training else torch.enable_grad():
            output = self.transformer(**encoded)
        
        # Use CLS token representation
        cls_output = output.last_hidden_state[:, 0, :]
        
        # Project to output dimension
        out = self.output_proj(cls_output)
        
        return out

class UncertaintyHead(nn.Module):
    """Uncertainty estimation head"""
    
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.log_var = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        mean = self.mean(x)
        log_var = self.log_var(x)
        # Ensure positive variance
        var = F.softplus(log_var) + 1e-6
        return mean, var

class StateOfTheArtPolymerPredictor(nn.Module):
    """Complete state-of-the-art model"""
    
    def __init__(self, feature_dim, gnn_output_dim=256, transformer_output_dim=256, 
                 hidden_dim=512):
        super().__init__()
        
        # Graph Neural Network
        self.gnn = AdvancedGNN(
            node_dim=150,
            edge_dim=10,
            hidden_dim=config.gnn_hidden_dim,
            output_dim=gnn_output_dim,
            n_layers=config.n_gnn_layers,
            n_heads=config.n_attention_heads,
            dropout=config.dropout
        )
        
        # Transformer for SMILES
        if config.use_transformer:
            self.transformer = PolymerTransformer(
                output_dim=transformer_output_dim
            )
        
        # Feature processing with layer normalization
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Fusion network
        fusion_dim = gnn_output_dim + hidden_dim // 2
        if config.use_transformer:
            fusion_dim += transformer_output_dim
        
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Task-specific heads with uncertainty
        self.heads = nn.ModuleDict()
        self.uncertainty_heads = nn.ModuleDict()
        
        for prop in config.properties:
            self.heads[prop] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            if config.use_uncertainty:
                self.uncertainty_heads[prop] = UncertaintyHead(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, graph_batch, features, smiles_list=None):
        # Normalize features
        features = self.feature_norm(features)
        
        # GNN encoding
        gnn_out = self.gnn(
            graph_batch.x, 
            graph_batch.edge_index, 
            graph_batch.edge_attr,
            graph_batch.batch
        )
        
        # Feature encoding
        feat_out = self.feature_net(features)
        
        # Combine representations
        representations = [gnn_out, feat_out]
        
        # Transformer encoding
        if config.use_transformer and smiles_list is not None:
            transformer_out = self.transformer(smiles_list)
            representations.append(transformer_out)
        
        # Fusion
        combined = torch.cat(representations, dim=1)
        combined = self.fusion_norm(combined)
        fusion_out = self.fusion(combined)
        
        # Predictions with uncertainty
        predictions = {}
        uncertainties = {}
        
        for prop in config.properties:
            predictions[prop] = self.heads[prop](fusion_out)
            
            if config.use_uncertainty:
                mean, var = self.uncertainty_heads[prop](fusion_out)
                predictions[prop] = mean
                uncertainties[prop] = var
        
        if config.use_uncertainty:
            return predictions, uncertainties
        else:
            return predictions

# ============================================================================
# DATASET AND TRAINING
# ============================================================================

class AdvancedPolymerDataset(Dataset):
    """Dataset for polymer prediction with all features"""
    
    def __init__(self, smiles, features, graphs, targets=None):
        self.smiles = smiles
        self.features = torch.FloatTensor(features)
        self.graphs = graphs
        self.targets = targets
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        item = {
            'smiles': self.smiles[idx],
            'features': self.features[idx],
            'graph': self.graphs[idx]
        }
        
        if self.targets is not None:
            item['targets'] = {}
            for prop in config.properties:
                if prop in self.targets.columns:
                    val = self.targets[prop].iloc[idx]
                    # Use NaN mask instead of placeholder
                    if pd.isna(val):
                        item['targets'][prop] = torch.FloatTensor([val])
                        item['targets'][f'{prop}_mask'] = torch.FloatTensor([0.0])
                    else:
                        item['targets'][prop] = torch.FloatTensor([val])
                        item['targets'][f'{prop}_mask'] = torch.FloatTensor([1.0])
        
        return item

def advanced_collate_fn(batch):
    """Custom collate function for advanced features"""
    smiles = [item['smiles'] for item in batch]
    features = torch.stack([item['features'] for item in batch])
    graphs = Batch.from_data_list([item['graph'] for item in batch])
    
    result = {
        'smiles': smiles,
        'features': features,
        'graphs': graphs
    }
    
    if 'targets' in batch[0]:
        targets = {}
        for prop in config.properties:
            if prop in batch[0]['targets']:
                # Stack values (including NaN)
                values = []
                masks = []
                for item in batch:
                    values.append(item['targets'][prop])
                    masks.append(item['targets'][f'{prop}_mask'])
                
                targets[prop] = torch.stack(values)
                targets[f'{prop}_mask'] = torch.stack(masks)
        
        result['targets'] = targets
    
    return result

def weighted_mae_loss_with_uncertainty(predictions, targets, uncertainties, weights):
    """Weighted MAE loss with uncertainty and missing value handling"""
    total_loss = 0
    losses = {}
    valid_props = 0
    
    for prop in config.properties:
        if prop in predictions and prop in targets:
            # Get mask for valid values
            mask = targets[f'{prop}_mask']
            
            if mask.sum() > 0:
                # Get valid predictions and targets
                valid_mask = mask.bool()
                valid_pred = predictions[prop][valid_mask]
                valid_target = targets[prop][valid_mask]
                
                # MAE loss
                mae = torch.abs(valid_pred - valid_target).mean()
                
                # Uncertainty loss (if available)
                if prop in uncertainties:
                    valid_uncertainty = uncertainties[prop][valid_mask]
                    # Negative log likelihood with uncertainty
                    nll = 0.5 * torch.log(valid_uncertainty) + \
                          0.5 * (valid_pred - valid_target)**2 / valid_uncertainty
                    uncertainty_loss = nll.mean()
                    
                    # Combined loss
                    prop_loss = mae + 0.1 * uncertainty_loss
                else:
                    prop_loss = mae
                
                losses[prop] = prop_loss.item()
                
                # Apply property weight
                weight = config.property_weights.get(prop, 1.0)
                total_loss += weight * prop_loss
                valid_props += weight
    
    # Normalize by total weight
    if valid_props > 0:
        total_loss = total_loss / valid_props
    else:
        # Return small non-zero loss to maintain gradients
        total_loss = torch.tensor(0.001, requires_grad=True, device=predictions[list(predictions.keys())[0]].device)
    
    return total_loss, losses

def train_one_epoch(model, loader, optimizer, scaler=None):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    batch_count = 0
    
    # Reset gradients
    optimizer.zero_grad()
    
    progress_bar = tqdm(loader, desc="Training")
    for i, batch in enumerate(progress_bar):
        features = batch['features'].to(config.device)
        graphs = batch['graphs'].to(config.device)
        targets = {k: v.to(config.device) for k, v in batch['targets'].items()}
        smiles = batch['smiles']
        
        # Forward pass
        if config.use_uncertainty:
            predictions, uncertainties = model(graphs, features, smiles)
        else:
            predictions = model(graphs, features, smiles)
            uncertainties = {}
        
        # Calculate loss
        loss, prop_losses = weighted_mae_loss_with_uncertainty(
            predictions, targets, uncertainties, config.property_weights
        )
        
        # Scale loss for gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (i + 1) % config.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss.item() * config.gradient_accumulation_steps
        batch_count += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item() * config.gradient_accumulation_steps,
            **{f'{k}_mae': v for k, v in prop_losses.items()}
        })
    
    return total_loss / max(batch_count, 1)

def validate(model, loader):
    """Validate model"""
    model.eval()
    total_loss = 0
    property_losses = defaultdict(float)
    property_counts = defaultdict(int)
    batch_count = 0
    
    all_predictions = defaultdict(list)
    all_targets = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            features = batch['features'].to(config.device)
            graphs = batch['graphs'].to(config.device)
            targets = {k: v.to(config.device) for k, v in batch['targets'].items()}
            smiles = batch['smiles']
            
            # Forward pass
            if config.use_uncertainty:
                predictions, uncertainties = model(graphs, features, smiles)
            else:
                predictions = model(graphs, features, smiles)
                uncertainties = {}
            
            # Calculate loss
            loss, prop_losses = weighted_mae_loss_with_uncertainty(
                predictions, targets, uncertainties, config.property_weights
            )
            
            if loss.item() > 0:
                total_loss += loss.item()
                batch_count += 1
                
                for prop, prop_loss in prop_losses.items():
                    property_losses[prop] += prop_loss
                    property_counts[prop] += 1
            
            # Store predictions for evaluation
            for prop in config.properties:
                if prop in predictions and f'{prop}_mask' in targets:
                    mask = targets[f'{prop}_mask'].bool()
                    if mask.sum() > 0:
                        all_predictions[prop].extend(predictions[prop][mask].cpu().numpy())
                        all_targets[prop].extend(targets[prop][mask].cpu().numpy())
    
    # Calculate average losses
    avg_loss = total_loss / max(batch_count, 1)
    avg_property_losses = {}
    
    for prop in property_losses:
        if property_counts[prop] > 0:
            avg_property_losses[prop] = property_losses[prop] / property_counts[prop]
    
    # Calculate overall MAE for each property
    property_mae = {}
    for prop in all_predictions:
        if len(all_predictions[prop]) > 0:
            mae = mean_absolute_error(all_targets[prop], all_predictions[prop])
            property_mae[prop] = mae
    
    return avg_loss, avg_property_losses, property_mae

# ============================================================================
# ML ENSEMBLE
# ============================================================================

def train_advanced_ml_ensemble(X_train, y_train, X_val, y_val, property_name):
    """Train advanced ML ensemble for a specific property"""
    models = {}
    
    # XGBoost with advanced parameters
    xgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'max_depth': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_weight': 5,
        'gamma': 0.1,
        'random_state': config.seed,
        'n_jobs': config.n_jobs,
        'tree_method': 'hist'
    }
    
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    models['xgb'] = xgb_model
    
    # LightGBM with advanced parameters
    lgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'num_leaves': 127,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_split_gain': 0.01,
        'min_child_weight': 0.001,
        'random_state': config.seed,
        'n_jobs': config.n_jobs,
        'metric': 'mae'
    }
    
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    models['lgb'] = lgb_model
    
    # CatBoost with advanced parameters
    cb_params = {
        'iterations': 2000,
        'learning_rate': 0.01,
        'depth': 10,
        'l2_leaf_reg': 3,
        'min_data_in_leaf': 20,
        'random_strength': 0.5,
        'bagging_temperature': 0.5,
        'od_type': 'Iter',
        'od_wait': 100,
        'random_seed': config.seed,
        'verbose': False
    }
    
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )
    models['cb'] = cb_model
    
    # Extra Trees for diversity
    et_model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=config.seed,
        n_jobs=config.n_jobs
    )
    et_model.fit(X_train, y_train)
    models['et'] = et_model
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=config.seed,
        n_jobs=config.n_jobs
    )
    rf_model.fit(X_train, y_train)
    models['rf'] = rf_model
    
    return models

def predict_with_ml_ensemble(models, X_test):
    """Make predictions with ML ensemble"""
    predictions = {}
    
    # Get predictions from each model
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
    
    # Weighted average based on typical performance
    weights = {
        'xgb': 0.3,
        'lgb': 0.3,
        'cb': 0.2,
        'et': 0.1,
        'rf': 0.1
    }
    
    # Calculate weighted prediction
    weighted_pred = np.zeros_like(predictions['xgb'])
    for name, pred in predictions.items():
        weighted_pred += weights.get(name, 0.1) * pred
    
    return weighted_pred, predictions

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def load_and_prepare_data():
    """Load and prepare all data"""
    print("\n📊 Loading data...")
    
    # Load main data
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    
    # Load supplement data
    supplement_dfs = []
    if os.path.exists(config.supplement_path):
        for file in sorted(os.listdir(config.supplement_path)):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(config.supplement_path, file))
                supplement_dfs.append(df)
                print(f"  Loaded {file}: {len(df)} samples")
    
    # Combine all training data
    if supplement_dfs:
        all_train_df = pd.concat([train_df] + supplement_dfs, ignore_index=True)
    else:
        all_train_df = train_df
    
    print(f"\nTotal training samples: {len(all_train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Show data availability
    print("\n📊 Data availability per property:")
    for prop in config.properties:
        if prop in all_train_df.columns:
            available = (~all_train_df[prop].isna()).sum()
            percentage = available / len(all_train_df) * 100
            print(f"  {prop}: {available}/{len(all_train_df)} ({percentage:.1f}%)")
    
    return all_train_df, test_df

def extract_all_features(df, feature_extractor, desc="Extracting features"):
    """Extract all features with progress bar"""
    features = []
    
    # Use multiprocessing for faster extraction
    with Pool(config.n_jobs) as pool:
        features = list(tqdm(
            pool.imap(feature_extractor.extract_features, df['SMILES'].values, chunksize=10),
            total=len(df),
            desc=desc
        ))
    
    return np.array(features)

def main():
    print("\n" + "="*70)
    print("🏆 NeurIPS 2025 - Open Polymer Prediction Challenge")
    print("💰 Prize: $1,000,000")
    print("🚀 STATE-OF-THE-ART IMPLEMENTATION")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Advanced Graph Neural Networks (Transformer, GAT, GIN)")
    print("  ✓ Pre-trained ChemBERTa transformer")
    print("  ✓ Mordred descriptors (1600+)")
    print("  ✓ 3D conformer features")
    print("  ✓ Uncertainty quantification")
    print("  ✓ Advanced ML ensemble (XGBoost, LightGBM, CatBoost, RF, ET)")
    print("  ✓ Weighted MAE for multi-task learning")
    print("  ✓ Handles missing values correctly")
    print("="*70)
    
    # Load data
    train_df, test_df = load_and_prepare_data()
    
    # Initialize feature extractor
    print("\n🔬 Initializing advanced feature extractor...")
    feature_extractor = AdvancedMolecularFeatureExtractor(
        use_mordred=config.use_mordred,
        use_3d=config.use_3d,
        cache_dir=config.cache_dir
    )
    
    # Extract features
    print("\n🧪 Extracting molecular features (this may take a while)...")
    
    # Check cache
    train_features_path = os.path.join(config.cache_dir, 'train_features_advanced.npy')
    test_features_path = os.path.join(config.cache_dir, 'test_features_advanced.npy')
    
    if os.path.exists(train_features_path) and os.path.exists(test_features_path) and config.use_cache:
        print("  Loading cached features...")
        train_features = np.load(train_features_path)
        test_features = np.load(test_features_path)
    else:
        train_features = extract_all_features(train_df, feature_extractor, "Train features")
        test_features = extract_all_features(test_df, feature_extractor, "Test features")
        
        # Save features
        np.save(train_features_path, train_features)
        np.save(test_features_path, test_features)
    
    print(f"  Feature shape: {train_features.shape}")
    
    # Convert to graphs
    print("\n🔗 Converting to advanced molecular graphs...")
    train_graphs = [mol_to_advanced_graph(smiles) for smiles in tqdm(train_df['SMILES'].values, desc="Train graphs")]
    test_graphs = [mol_to_advanced_graph(smiles) for smiles in tqdm(test_df['SMILES'].values, desc="Test graphs")]
    
    # Feature preprocessing
    print("\n🎯 Advanced feature preprocessing...")
    
    # Remove low variance features
    selector = VarianceThreshold(threshold=0.0001)
    train_features_selected = selector.fit_transform(train_features)
    test_features_selected = selector.transform(test_features)
    print(f"  After variance threshold: {train_features_selected.shape[1]} features")
    
    # Apply quantile transformation for robustness
    quantile_transformer = QuantileTransformer(
        n_quantiles=min(1000, len(train_df)),
        output_distribution='normal',
        random_state=config.seed
    )
    train_features_transformed = quantile_transformer.fit_transform(train_features_selected)
    test_features_transformed = quantile_transformer.transform(test_features_selected)
    
    # Feature selection based on mutual information
    print("  Selecting top features by mutual information...")
    
    # For feature selection, use FFV as it has most data
    ffv_mask = ~train_df['FFV'].isna()
    if ffv_mask.sum() > 100:
        mi_selector = SelectKBest(mutual_info_regression, k=min(5000, train_features_transformed.shape[1]))
        mi_selector.fit(train_features_transformed[ffv_mask], train_df['FFV'][ffv_mask])
        train_features_final = mi_selector.transform(train_features_transformed)
        test_features_final = mi_selector.transform(test_features_transformed)
    else:
        train_features_final = train_features_transformed
        test_features_final = test_features_transformed
    
    print(f"  Final feature shape: {train_features_final.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_final)
    test_features_scaled = scaler.transform(test_features_final)
    
    # Prepare targets
    y_train = train_df[config.properties]
    
    # K-Fold Cross Validation
    print(f"\n🔄 Starting {config.n_folds}-fold cross-validation...")
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    
    # Storage for predictions
    all_oof_predictions = {prop: np.full(len(train_df), np.nan) for prop in config.properties}
    all_test_predictions = {prop: [] for prop in config.properties}
    all_test_predictions_ml = {prop: [] for prop in config.properties}
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features_scaled)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold+1}/{config.n_folds}")
        print(f"{'='*70}")
        
        # Split data
        X_train_fold = train_features_scaled[train_idx]
        X_val_fold = train_features_scaled[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        # Create datasets
        train_dataset = AdvancedPolymerDataset(
            train_df['SMILES'].iloc[train_idx].values,
            X_train_fold,
            [train_graphs[i] for i in train_idx],
            y_train_fold
        )
        
        val_dataset = AdvancedPolymerDataset(
            train_df['SMILES'].iloc[val_idx].values,
            X_val_fold,
            [train_graphs[i] for i in val_idx],
            y_val_fold
        )
        
        test_dataset = AdvancedPolymerDataset(
            test_df['SMILES'].values,
            test_features_scaled,
            test_graphs
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=advanced_collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            collate_fn=advanced_collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            collate_fn=advanced_collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        # Initialize model
        print("\n🤖 Initializing state-of-the-art model...")
        model = StateOfTheArtPolymerPredictor(
            feature_dim=X_train_fold.shape[1],
            gnn_output_dim=256,
            transformer_output_dim=256,
            hidden_dim=config.hidden_dim
        ).to(config.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch_metrics = {}
        
        for epoch in range(config.epochs):
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer)
            
            # Validate
            val_loss, val_prop_losses, val_mae = validate(model, val_loader)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Print metrics
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print("  Val MAE by property:")
            for prop, mae in val_mae.items():
                print(f"    {prop}: {mae:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch_metrics = val_mae.copy()
                
                # Save checkpoint
                checkpoint = {
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'scaler_state': scaler,
                    'selector_state': selector,
                    'quantile_transformer_state': quantile_transformer
                }
                torch.save(checkpoint, f"{config.checkpoint_dir}/fold_{fold}_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model
        checkpoint = torch.load(f"{config.checkpoint_dir}/fold_{fold}_best.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Store fold metrics
        fold_scores.append(best_epoch_metrics)
        
        # Get validation predictions
        print("\n📊 Getting validation predictions...")
        model.eval()
        val_predictions = {prop: [] for prop in config.properties}
        val_indices = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(config.device)
                graphs = batch['graphs'].to(config.device)
                smiles = batch['smiles']
                
                if config.use_uncertainty:
                    predictions, _ = model(graphs, features, smiles)
                else:
                    predictions = model(graphs, features, smiles)
                
                # Get batch indices
                batch_size = features.shape[0]
                start_idx = len(val_indices)
                val_indices.extend(val_idx[start_idx:start_idx + batch_size])
                
                for prop in config.properties:
                    if prop in predictions:
                        val_predictions[prop].extend(predictions[prop].cpu().numpy().flatten())
        
        # Store OOF predictions
        for prop in config.properties:
            if val_predictions[prop]:
                all_oof_predictions[prop][val_indices] = val_predictions[prop][:len(val_indices)]
        
        # Train ML ensemble for properties with enough data
        print("\n🤖 Training ML ensemble...")
        ml_models = {}
        
        for prop in config.properties:
            if prop in y_train_fold.columns:
                # Check if we have enough data
                train_mask = ~y_train_fold[prop].isna()
                val_mask = ~y_val_fold[prop].isna()
                
                if train_mask.sum() > 100 and val_mask.sum() > 20:
                    print(f"  Training ensemble for {prop}...")
                    
                    X_train_prop = X_train_fold[train_mask]
                    y_train_prop = y_train_fold[prop][train_mask]
                    X_val_prop = X_val_fold[val_mask]
                    y_val_prop = y_val_fold[prop][val_mask]
                    
                    ml_models[prop] = train_advanced_ml_ensemble(
                        X_train_prop, y_train_prop,
                        X_val_prop, y_val_prop,
                        prop
                    )
                    
                    # Validate ensemble
                    ensemble_pred, _ = predict_with_ml_ensemble(ml_models[prop], X_val_prop)
                    ensemble_mae = mean_absolute_error(y_val_prop, ensemble_pred)
                    print(f"    Ensemble MAE: {ensemble_mae:.4f}")
        
        # Get test predictions
        print("\n📈 Getting test predictions...")
        test_predictions_nn = {prop: [] for prop in config.properties}
        test_predictions_ml = {prop: [] for prop in config.properties}
        
        # Neural network predictions
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(config.device)
                graphs = batch['graphs'].to(config.device)
                smiles = batch['smiles']
                
                if config.use_uncertainty:
                    predictions, uncertainties = model(graphs, features, smiles)
                else:
                    predictions = model(graphs, features, smiles)
                
                for prop in config.properties:
                    if prop in predictions:
                        test_predictions_nn[prop].extend(predictions[prop].cpu().numpy().flatten())
        
        # ML ensemble predictions
        for prop in ml_models:
            ensemble_pred, _ = predict_with_ml_ensemble(ml_models[prop], test_features_scaled)
            test_predictions_ml[prop] = ensemble_pred
        
        # Store predictions
        for prop in config.properties:
            if prop in test_predictions_nn:
                all_test_predictions[prop].append(np.array(test_predictions_nn[prop]))
            if prop in test_predictions_ml:
                all_test_predictions_ml[prop].append(np.array(test_predictions_ml[prop]))
    
    # Calculate OOF scores
    print("\n📊 Out-of-Fold scores:")
    oof_scores = {}
    for prop in config.properties:
        if prop in y_train.columns:
            mask = ~y_train[prop].isna()
            if mask.sum() > 0:
                oof_pred = all_oof_predictions[prop][mask]
                oof_true = y_train[prop][mask]
                # Remove any remaining NaN in predictions
                valid_mask = ~np.isnan(oof_pred)
                if valid_mask.sum() > 0:
                    mae = mean_absolute_error(oof_true[valid_mask], oof_pred[valid_mask])
                    oof_scores[prop] = mae
                    print(f"  {prop}: MAE = {mae:.4f}")
    
    # Average test predictions
    print("\n📊 Averaging test predictions across folds...")
    final_predictions = {}
    
    for prop in config.properties:
        nn_preds = []
        ml_preds = []
        
        # Collect all predictions
        if prop in all_test_predictions:
            nn_preds = [p for p in all_test_predictions[prop] if len(p) > 0]
        if prop in all_test_predictions_ml:
            ml_preds = [p for p in all_test_predictions_ml[prop] if len(p) > 0]
        
        # Combine predictions
        if nn_preds and ml_preds:
            # Average neural network predictions
            nn_avg = np.mean(nn_preds, axis=0)
            # Average ML predictions
            ml_avg = np.mean(ml_preds, axis=0)
            # Weighted combination (70% NN, 30% ML for properties with less data)
            if prop == 'FFV':
                # FFV has most data, trust ML more
                final_predictions[prop] = 0.5 * nn_avg + 0.5 * ml_avg
            else:
                # Other properties have less data, trust NN more
                final_predictions[prop] = 0.7 * nn_avg + 0.3 * ml_avg
        elif nn_preds:
            final_predictions[prop] = np.mean(nn_preds, axis=0)
        elif ml_preds:
            final_predictions[prop] = np.mean(ml_preds, axis=0)
        else:
            # No predictions available for this property
            print(f"  No predictions available for {prop}")
            final_predictions[prop] = np.array([])
    
    # Use intelligent predictions instead of simple mean fallbacks
    print("\n🧠 Applying intelligent imputation strategies...")
    
    # Apply intelligent predictions
    final_predictions = create_intelligent_predictions(
        test_df=test_df,
        train_df=train_df,
        base_predictions=final_predictions,
        features_train=train_features_scaled,
        features_test=test_features_scaled
    )
    
    # Create submission
    print("\n📝 Creating submission...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        **final_predictions
    })
    
    # Final check for NaN values
    for prop in config.properties:
        if prop in submission.columns:
            nan_count = submission[prop].isna().sum()
            if nan_count > 0:
                print(f"  Warning: {nan_count} NaN values in {prop}")
                # This should not happen with intelligent imputation
                print(f"  Critical error: Intelligent imputation failed for {prop}")
    
    # Save submission
    submission.to_csv(config.submission_path, index=False)
    print(f"\n✅ Submission saved to {config.submission_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nFold scores:")
    for i, scores in enumerate(fold_scores):
        print(f"\nFold {i+1}:")
        for prop, score in scores.items():
            print(f"  {prop}: {score:.4f}")
    
    print("\nFinal OOF scores:")
    for prop, score in oof_scores.items():
        print(f"  {prop}: {score:.4f}")
    
    # Calculate weighted average based on data availability
    if oof_scores:
        weights_sum = 0
        weighted_score = 0
        for prop, score in oof_scores.items():
            weight = config.property_weights.get(prop, 1.0)
            weighted_score += weight * score
            weights_sum += weight
        
        if weights_sum > 0:
            final_score = weighted_score / weights_sum
            print(f"\nWeighted average MAE: {final_score:.4f}")
    
    print("\n🏁 Training complete!")
    print("Good luck in the competition! 🍀")

if __name__ == "__main__":
    main()