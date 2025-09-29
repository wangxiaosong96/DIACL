
# DIACL

Open source for "Disentangled Contrastive Learning with Dynamic Intent Adaptation for Unveiling Gene-Drug Associations"
![Disentangled Contrastive Learning with Dynamic Intent Adaptation for Unveiling Gene-Drug Associations](https://github.com/wangxiaosong96/DIACL/blob/main/DIACL.png)


# Environment
The code is implemented in Python 3.8.13. Key dependencies include:

Python 3.8.13

numpy == 1.22.3

pytorch == 1.11.0 (GPU version)

torch-scatter == 2.0.9

torch-sparse == 0.6.14

scipy == 1.9.3

# Dataset

We evaluated our model using three publicly available gene-drug interaction datasets:

| Dataset | # Genes | # Drugs | # Interactions |
|---------|----------|---------|----------------|
| DGIdb | 3,021 | 11,476 | 34,903 |
| ChEMBL | 1,555 | 4,418 | 12,075 |
| Guide To Pharmacology | 1,694 | 8,161 | 15,793 |

> **Note:** The validation set was used solely for hyperparameter tuning. For all datasets, the validation split was merged back into the training set during the final model training.

## Usage

### Training Commands

Train the model on each dataset using the following commands. The model runs for a fixed number of epochs, and parameters from the final epoch are used for testing.

**DGIdb**
```bash
python main.py --dataset dgidb --epoch 150
```

**ChEMBL**
```bash
python main.py --dataset chembl --epoch 100
```

**Guide To Pharmacology**
```bash
python main.py --dataset gtopharm --epoch 100
```

### Additional Options

For advanced usage and a complete list of available arguments:
```bash
python main.py --help
```

---

# Citation
X Wang, Y Liu, Q Wang, et al. Disentangled Contrastive Learning with Dynamic Intent Adaptation for Unveiling Gene-Drug Associations[J]. Briefings in Bioinformatics, 2025.
