# CS4641-Project

`/code/`: Contains the Jupyter notebooks used to pre-process data and train models  
`/code/Audio Augmentation.ipynb`: Notebook for taking in audio files, applying augmentation, and extracting mean and variance of several audio features  
`/code/Dimensionality Reduction.ipynb`: Notebook in which PCA and LDA are applied to augmented audio features  
`/code/LDA.ipynb`: Notebook in which LDA-reduced features are used to train/validate/test SVM, RFC, and ANN  
`/code/Original.ipynb`: Notebook in which non-reduced features are used to train/validate/test SVM, RFC, and ANN  
`/code/PCA.ipynb`: Notebook in which PCA-reduced features are used to train/validate/test SVM, RFC, and ANN  
`/code/RNN Training.ipynb`: Notebook in which sequential data features are used to train/validate/test RNN  
`/code/Sequential Data Extraction.ipynb`: Notebook in which augmented audio files are sampled to extract sequential audio features  
`/data/`: Contains csv files of audio features produced by each pre-processing method  
`/data/LDA_format.csv`: Audio feature data after LDA is applied to original data  
`/data/Original_format.csv`: Original (no dimension reduction) audio feature data produced from audio augmentation  
`/data/PCA_format.csv`: Audio feature data after PCA is applied to original data  
`/data/Squential_format.md`: Markdown file with link to sequential audio feature data produced from audio sampling  