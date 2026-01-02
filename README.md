# Transfer Learning for Tree Classification in Ghana

Distinguishing natural and agricultural tree systems is critical for monitoring ecosystem services, commodity-driven deforestation, and restoration progress. From a remote sensing standpoint, this task is particularly challenging because: 
* Certain trees exhibit high spectral similarity
* Highly heterogeneous, smallholder agricultural landscapes require a small minimum mapping unit to detect differences
* Regions with persistent cloud cover and atmospheric haze reduce optical image quality

The distinction is critical for restoration monitoring applications, as gains in tree cover cannot be meaningfully assessed
without understanding whether they result from a successful restoration intervention or agricultural expansion. 

## Research Summary
This project applies a transfer learning approach to classify tree-based land use systems from satellite imagery. We leverage spatial embeddings extracted from a high-performing convolutional neural network originally trained for tree cover mapping ([Brandt et al., 2023](https://github.com/wri/sentinel-tree-cover)) and repurpose them for land use classification.

We train a CatBoost classifier using a combination of Sentinel-1 and Sentinel-2 imagery, gray-level co-occurrence matrix (GLCM) texture features, and extracted spatial embeddings to classify four land use classes: **natural**, **agroforestry**, **monoculture**, and **other (background)**. Through comparative modeling and feature selection, we demonstrate consistent performance gains from incorporating both transfer-learned features and texture information.

In collaboration with Ghana’s Environmental Protection Agency, the method is demonstrated across 26 priority districts, resulting in a 10-meter resolution land use map for 2020. Overall, the findings suggest that spatial embeddings learned for tree detection retain meaningful information about land use structure, offering a scalable pathway for broader monitoring of natural and agricultural tree systems.

**Download the paper:** [Technical Note](https://www.wri.org/research/transfer-learning-detect-natural-monoculture-and-agroforestry-tree-based-systems-ghana)  
**View the data:** [Ghana EPA Monitoring Portal](https://environmental-protection-agency-epa-ghana.hub.arcgis.com/pages/generalmap) (_toggle on WRI Land Use_)

![Pixel-based Land Use Classification Results](images/fig5.jpg)

**Suggested citation:**  
Ertel, J., J. Brandt, R. Rognstad, and E. Glen (2025). *Transfer learning to detect natural, monoculture, and agroforestry tree-based systems in Ghana using remote sensing*. Technical Note. Washington, DC: World Resources Institute. doi:10.46830/writn.24.00030

---
## Machine Learning Pipeline
The predictive features for the classification task include Sentinel-1 and Sentinel-2 imagery, spatial embeddings and texture features. Texture features were derived from Sentinel-2 imagery using a GLCM analysis method. The figure below shows the machine learning pipeline, including pre- and post-processing steps.

![ML Workflow](images/ml_pipeline.jpg)

### DVC Setup & Directory Structure

The `src` directory is designed to integrate with Data Version Control (DVC) to ensure reproducibility and efficient management of the project's machine learning pipeline.  

This directory contains modular scripts and functions organized to support a DVC workflow. Each script performs a specific task within the pipeline, such as data preparation, model training, or evaluation. The pipeline stages are connected through DVC.

#### Why DVC?
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

### Using DVC Pipelines
The pipeline is defined in the `dvc.yaml` file, with dependencies, parameters, and outputs explicitly stated for each stage.

### Parameters Management
Pipeline parameters are defined in `params.yaml`. This file centralizes hyperparameters and configuration options for each stage of the pipeline. Update parameters as needed, and rerun the pipeline using `dvc repro` to propagate changes.

---

### Additional Resources
- [Tropical Tree Cover](https://github.com/wri/sentinel-tree-cover)
- [CatBoost Documentation](https://catboost.ai/docs/en/)
- [DVC Documentation](https://dvc.org/doc)