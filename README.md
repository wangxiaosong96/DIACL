
# DIACL

Open source for "Disentangled Contrastive Learning with Dynamic Intent Adaptation for Unveiling Gene-Drug Associations"

# Environment
The codes are written in Python 3.8.13 with the following dependencies.

numpy == 1.22.3
pytorch == 1.11.0 (GPU version)
torch-scatter == 2.0.9
torch-sparse == 0.6.14
scipy == 1.9.3

Dataset
We evaluated our model using three publicly available gene–drug interaction datasets: DGIdb, ChEMBL, and Guide To Pharmacology. Table 1 summarizes their basic statistics.

# Table 1. Statistics of the evaluated datasets.

Dataset	# Genes	# Drugs	# Interactions
DGIdb	3,021	11,476	34,903
ChEMBL	1,555	4,418	12,075
Guide To Pharmacology	1,694	8,161	15,793
The validation set was used solely for hyperparameter tuning. For all datasets, the validation split was merged back into the training set during the final model training.


# Citation
X Wang, Y Liu, Q Wang, et al. Disentangled Contrastive Learning with Dynamic Intent Adaptation for Unveiling Gene-Drug Associations[J]. Briefings in Bioinformatics, 2025.
