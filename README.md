# Transfer Learning for Tree Classification in Ghana

Distinguishing natural and agricultural tree systems is critical for monitoring ecosystem services, commodity-driven deforestation, and restoration progress. From a remote sensing standpoint, this task is particularly challenging for 3 reasons: 
1. Certain trees exhibit high spectral similarity
2. Highly heterogeneous, smallholder agricultural landscapes require a small minimum mapping unit to detect differences
3. Regions with persistent cloud cover and atmospheric haze reduce optical image quality

The distinction is critical for restoration monitoring applications, as gains in tree cover cannot be meaningfully assessed
without understanding whether they result from a successful restoration intervention or agricultural expansion. 

## Research Summary
This project applies a transfer learning approach to classify tree-based land use systems from satellite imagery. We leverage spatial embeddings extracted from a high-performing convolutional neural network originally trained for tree cover mapping ([Brandt et al. (2023)](https://github.com/wri/sentinel-tree-cover)) and repurpose them for land use classification.

We train a CatBoost classifier using a combination of Sentinel-1 and Sentinel-2 imagery, gray-level co-occurrence matrix (GLCM) texture features, and extracted spatial embeddings to classify four land use classes: **natural**, **agroforestry**, **monoculture**, and **other (background)**. Through comparative modeling and feature selection, we demonstrate consistent performance gains from incorporating both transfer-learned features and texture information.

In collaboration with Ghana’s Environmental Protection Agency, the method is demonstrated across 26 priority districts, resulting in a 10-meter resolution land use map for 2020. Overall, the findings suggest that spatial embeddings learned for tree detection retain meaningful information about land use structure, offering a scalable pathway for broader monitoring of natural and agricultural tree systems.

**Download the paper:** [WRI Technical Note](https://www.wri.org/research/transfer-learning-detect-natural-monoculture-and-agroforestry-tree-based-systems-ghana)
**View the data:** [Ghana EPA Restoration Monitoring Portal](https://environmental-protection-agency-epa-ghana.hub.arcgis.com/pages/generalmap) (_toggle on WRI Land Use_)

![Pixel-based Land Use Classification Results](images/fig5.jpg)

**Suggested citation:**  
Ertel, J., J. Brandt, R. Rognstad, and E. Glen (2025). *Transfer learning to detect natural, monoculture, and agroforestry tree-based systems in Ghana using remote sensing*. Technical Note. Washington, DC: World Resources Institute. doi:10.46830/writn.24.00030

---
## Machine Learning Pipeline
![ML Workflow](images/ml_pipeline.jpg)

### DVC Setup & Directory Structure

This section provides a guide to understand the structure of the `src` directory. The directory is designed to integrate with Data Version Control (DVC) to ensure reproducibility and efficient management of the project's machine learning pipeline.  

This directory contains modular scripts and functions organized to support a DVC workflow. Each script performs a specific task within the pipeline, such as data preparation, model training, or evaluation. The pipeline stages are connected through DVC.

#### Why DVC?
- Modular and reusable scripts.
- Clear separation of pipeline stages.
- Improved tracking for dependencies, outputs, and metadata.
- Compatible with YAML-based pipeline configurations.

---

### Directory Structure
```
/src
├── stage_load_data.py          # Scripts for ingesting raw data
├── stage_prep_features.py      # Scripts for data cleaning, transformation, and feature engineering
├── stage_train_model.py        # Script to train machine learning models
├── stage_select_and_tune.py    # Script to perform hyperparameter tuning and feature selection
├── stage_evaluate_model.py     # Script to evaluate model performance
└── transfer_learning.py        # Script for running inference
```

---

### Setting Up the Environment
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/wri/transfer-trees.git
   cd transfer-trees
   ```

2. **Install Dependencies:**
   Create a virtual environment and install dependencies from `requirements.txt`:
   ```bash
   python -m venv env
   source env/bin/activate  
   pip install -r requirements.txt
   ```

3. **Install DVC:**
   Ensure that DVC is installed for managing the pipeline:
   ```bash
   pip install dvc
   ```

4. **Initialize DVC:**
   If DVC is not already initialized in the repository:
   ```bash
   dvc init
   ```

---

## Using DVC Pipelines
The pipeline is defined in the `dvc.yaml` file, with dependencies, parameters, and outputs explicitly stated for each stage.

## Parameters Management
Pipeline parameters are defined in `params.yaml`. This file centralizes hyperparameters and configuration options for each stage of the pipeline. Update parameters as needed, and rerun the pipeline using `dvc repro` to propagate changes.

---

## Additional Resources
- [DVC Documentation](https://dvc.org/doc)