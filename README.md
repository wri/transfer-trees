# Transfer Learning for Tree Classification in Ghana

## Problem Statement
Distinguishing natural and agricultural tree systems is critical for monitoring ecosystem services, commodity-driven deforestation, and restoration progress. In Ghana, this task is particularly challenging due to (1) high spectral similarity between certain trees, (2) the small minimum mapping units required to capture heterogeneous smallholder agricultural landscapes, and (3) persistent cloud cover and atmospheric haze that limit optical image quality.

## Summary
This project applies a transfer learning approach to classify tree-based land use systems from satellite imagery. We leverage spatial embeddings extracted from a high-performing convolutional neural network originally trained for tree cover mapping and repurpose them for land use classification.

We train a CatBoost classifier using a combination of Sentinel-1 and Sentinel-2 imagery, gray-level co-occurrence matrix (GLCM) texture features, and extracted spatial embeddings to classify four land use classes: **natural**, **agroforestry**, **monoculture**, and **other (background)**. Through comparative modeling and feature selection, we demonstrate consistent performance gains from incorporating both transfer-learned features and texture information.

Building on the work of [Brandt et al. (2023)](https://github.com/wri/sentinel-tree-cover), this research explores whether learned spatial representations from a tree cover model can be reused to distinguish tree-based systems. In collaboration with Ghana’s Environmental Protection Agency, the method is demonstrated across 26 priority districts, resulting in a 10-meter resolution land use map for 2020.

Overall, the findings suggest that spatial embeddings learned for tree detection retain meaningful information about land use structure, offering a scalable pathway for broader monitoring of natural and agricultural tree systems.

**Download the paper:**  
[WRI Technical Note](https://www.wri.org/research/transfer-learning-detect-natural-monoculture-and-agroforestry-tree-based-systems-ghana)

**View the data:**  
[Ghana EPA Restoration Monitoring Portal](https://environmental-protection-agency-epa-ghana.hub.arcgis.com/pages/generalmap)  
(_toggle on WRI Land Use_)

**Suggested citation:**  
Ertel, J., J. Brandt, R. Rognstad, and E. Glen (2025). *Transfer learning to detect natural, monoculture, and agroforestry tree-based systems in Ghana using remote sensing*. Technical Note. Washington, DC: World Resources Institute. doi:10.46830/writn.24.00030

![Pixel-based Land Use Classification Results](images/fig5.jpg)


## Repository Organization
```
├── LICENSE.txt
├── README.md                                                  
├── Dockerfile                                      
├── params.yaml                      
├── config.yaml                      
├── dvc.yaml 
├── dvc.lock 
├── envs/                       
├── src                                 <- Source code for use in this project.
│   ├── __init__.py                        
│   ├── stage_load_data.py          
│   ├── stage_prep_features.py      
│   ├── stage_select_and_tune.py    
│   ├── stage_train_model.py        
│   ├── stage_evaluate_model.py     
│   ├── transfer_learning.py        
│   │
│   ├── load_data                       <- Scripts to download or generate data
│   │   ├── __init__.py            
│   │   └── s3_download.py           
│   │
│   ├── features                        <- Scripts to import and prepare modeling inputs
│   │   ├── __init__.py             
│   │   ├── PlantationsData.py      
│   │   ├── create_xy.py            
│   │   ├── feature_selection.py    
│   │   ├── texture_analysis.py    
│   │   ├── slow_glcm.py            
│   │   └── fast_glcm.py            
│   │    
│   ├── model                           <- Scripts to train models, select features, tune
│   │   ├── __init__.py             
│   │   ├── train.py                   
│   │   └── tune.py               
│   │    
│   ├── evaluation                      <- Graphics and figures from model evaluation
│   │   ├── confusion_matrix_data.csv       
│   │   ├── confusion_matrix.png            
│   │   └── validation_visuals.py           
│   │
│   └── utils                           <- Scripts for utility functions
│       ├── __init__.py             
│       ├── cloud_removal.py         
│       ├── interpolation.py          
│       ├── proximal_steps.py        
│       ├── indices.py                
│       ├── logs.py                   
│       ├── preprocessing.py         
│       ├── validate_io.py          
│       ├── quick_viz.py             
│       └── mosaic.py               
│
├── notebooks                           <- Jupyter notebooks                         
│   ├── analyses         
│   ├── features     
│   ├── modeling      
│   └── training_data
│
├── .gitignore                     
├── .dockerignore                  
└── .dvcignore                   
```
## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

[images/fig5.jpg]: images/fig