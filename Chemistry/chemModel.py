import os
import sys
import warnings
import multiprocessing
import multiprocessing.pool

# 1. THE PYTHON 3.13 MULTIPROCESSING PATCH
# We must do this before importing DeepChem
if not hasattr(multiprocessing, 'dummy'):
    multiprocessing.dummy = multiprocessing.pool

# 2. SILENCE NOISY WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)

import deepchem as dc
from rdkit import Chem

def run_training():
    print("🚀 Starting Molecular AI Training...")

    # 3. LOAD DATASET
    # Using CircularFingerprint (ECFP) is the most stable method for your setup
    featurizer = dc.feat.CircularFingerprint(size=2048, radius=2)
    
    print("📂 Loading Delaney Solubility Dataset...")
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=featurizer)
    train_dataset, valid_dataset, test_dataset = datasets

    # 4. INITIALIZE THE NEURAL NETWORK (MultitaskRegressor)
    # This model learns to map 2048-bit patterns to solubility numbers
    model = dc.models.MultitaskRegressor(
        n_tasks=1, 
        n_features=2048, 
        layer_sizes=[1000, 500], 
        dropouts=0.2
    )

    # 5. TRAIN
    print("🏋️ Training (Phase 1: Adjusting Weights)...")
    model.fit(train_dataset, nb_epoch=50)

    # 6. EVALUATE
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    test_scores = model.evaluate(test_dataset, [metric], transformers)
    print(f"\n✅ Training Complete! Accuracy (R2 Score): {test_scores['pearson_r2_score']:.4f}")

    # 7. PREDICT
    print("\n🔮 Testing on new molecules:")
    smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O'] # Ethanol, Benzene, Acetic Acid
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    features = featurizer.featurize(mols)
    predictions = model.predict_on_batch(features)

    for i, s in enumerate(smiles_list):
        print(f"   Structure: {s:10} | Predicted Solubility: {predictions[i][0]:.4f}")

# THIS IS THE CRITICAL LINE FOR WINDOWS
if __name__ == "__main__":
    # Ensure multiprocessing works correctly on Windows
    multiprocessing.freeze_support()
    run_training()