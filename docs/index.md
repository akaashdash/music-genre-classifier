# Music Genre Categorization

## Team
Dorsa Ajami - dajami3@gatech.edu  
Ariya Foroshani - aforoshani3@gatech.edu  
Stella Hansen - ehansen35@gatech.edu  
Nikan Hassanipak - nhassanipak3@gatech.edu  
Akaash Dash - adash37@gatech.edu

**Mentor:** Mohammad Taher - mtaher3@gatech.edu

## Introduction/Background
Our project will cover genre classification in the music industry, based on the features extracted from the audio files of songs with explicit genre labels. The [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) contains the following for each song in the 1000-song dataset of 10 genres:
- Raw audio files (.wav) 
- Mel-Spectrogram representations 
- .csv containing features extracted from the audio files, and the song's genre label (57 features per song).

Research for genre classification in the music industry has seen many major publications, with one approach highlighting the importance of combining preprocessing techniques and feature extraction (e.g. normalization, MFCCs, UWT, and LPCCs) with various traditional ML algorithms (KNN, SVM, ANN) [1]. More recent research emphasizes the transition from traditional feature engineering towards end-to-end learning models, along with the need for models that can be generalized across a variety of musical pieces, thereby introducing deep learning approaches such as CNN and RNN [2,3]. 

## Problem Definition
Manual tagging of songs by genre is flawed, but a new tagging system that uses machine learning to classify the genre of songs could address the following shortcomings: 
- **Scalability/efficiency:** Automating genre classification would save music distribution companies time and resources.
- **Consistency and objectivity:** Humans are prone to bias, whereas machine learning would apply consistent criteria to all songs. 
- **Genre equality:** Machine learning models can impartially identify emerging genres, giving equal visibility to upcoming artists not fitting traditional genres. 
- **Discovery/Recommendations:** More accurate categorization of music enhances music recommendation systems.

## Methods
### Data Preprocessing Methods:
- PCA: To mitigate the risk of overfitting (due to 57 features), we will produce a reduced-dimension version of the data.
- LDA: Provides an alternative dimensionality reduction method that is suited for maximizing class separation.
- Audio augmentation: By applying pitch/tempo shifting, we can generate augmented versions of the songs for the ML models, improving performance on unseen data.


### ML Algorithms/Models:
- SVM: Popular for audio classification tasks, particularly due to its performance in high-dimensions.
- RFC: Provides the benefits for classification tasks that decision trees offer, whilst minimizing overfitting.
- ANN: Deep learning architecture allows for the non-linear relationships between audio features and genres to be captured. Would also indicate if more complex deep learning models (CNN/RNN) are feasible for this problem and dataset.
- RNN: Cited in the literature as being at the forefront of audio classification tasks (especially when features are extracted from the audio file).

\* Note that during the midterm report we decided to pivot from a logistic regression model to an ANN model due to the fact that we already proved that non-deep learning methods (SVM/RFC) can learn our data and complete the classification task to a high and satisfactory accuracy. We would therefore like to pursue more deep learning approaches that can capture the complexity of our audio features to a higher degree. 

### Preprocessing Method 1: Audio Augmentation

The initial preprocessing method that we will be applying to our data is audio augmentation, which is a beneficial technique frequently used in tasks such as speech recognition, sound classification, and music analysis. The goal of audio augmentation is essentially to apply different augmentations to the original versions of the songs, such that we generate more data points for our models to be trained on. We implemented this method through the augment_audio function (found in "Audio Augmentation.ipynb") which takes in the time series of the original song, and returns three augmented versions of the song: one with random noise added on top, one with a time stretch, and one with a pitch shift. When we called this function, we used the parameters noise_factor=0.005 (minimal random noise is added), stretch_factor=0.8 (sped up version is returned), and n_steps=-1 (lower pitched version is returned). Since we only have 1,000 original songs, this preprocessing technique helps improve our models’ performances to unseen data (due to the fact that we now have 4,000 total data points, and the minor differences applied to generate the new data points will help improve classification within the validation set whilst also reducing the risk of overfitting to the training set, since the models are exposed to a wider array of data points).

It should be noted that this audio augmentation method will require us to be very careful when performing our train-test split for our model evaluations to ensure that no data leakage occurs. If two versions of the same song are allowed to be assigned to different sides of the train-test split, the models will have falsely accurate performances, since they will recognize that the training set has a very similar song to the one being evaluated within the testing set. This is why within our code, we use the songs’ predetermined title formatting to our advantage. Each genre has 100 songs, labeled in the format genre.song_number.wav, where song_number is a 5 digit value ranging from 00000 to 00099. We can therefore perform a manual five fold cross-validation, in which the song_numbers act as the index of the train-test split. So for fold one, songs with numbers from 00000 to 00019 will be used as the validation set, whilst songs with numbers from 00020 to 00099 are used as the training set. Not only does this ensure that all four versions of a song remain on the same side of the train-test split for any given fold, but it also implements stratification by making it such that each validation set uses the same number of songs from each genre as its support (ensuring that the evaluation metrics provide a more conclusive and accurate picture of the true performance of the model).

### Preprocessing Method 2: Principal Component Analysis (PCA)

The first dimensionality reduction technique used was PCA. PCA is an unsupervised method that does not use class labels and identifies the principal components of the dataset that capture the maximum variance. This process involves shifting the data onto a new z-space and selecting the top principal components that preserve the largest amount of the variance from the dataset. In our case, PCA was useful given the high dimensionality of our dataset, namely 57 features per sample, as it allowed us to reduce the risk of overfitting our model and reduces the complexity.  

To implement PCA on our data, we used the ‘PCA’ module from the ‘sklearn.decomposition’ package. We wanted to capture a large portion of the variance while still reducing the number of features, and therefore we chose to maintain 95% of the variance of the original dataset which translated to 34 principal components. Note that PCA assumes the directions with the largest variances in our z-space are most informative, which does not necessarily accomplish the best results when the objective is distinguishing between different music genres. This is an area where supervised learning algorithms such as LDA are better for maximizing class separability which is what we see when comparing their results for both SVM and RFC.

### Preprocessing Method 3: Linear Discriminant Analysis (LDA)

While PCA is a very useful dimensionality reduction technique, we wanted to take advantage of the labeled nature of our data during the pre-processing, which is why we also used Linear Discriminant Analysis (LDA) as an alternative dimensionality reduction technique. Unlike PCA, (no class labels required) which shifts the data onto a new z-space that maximizes the amount of variance captured (which may potentially aid in classification tasks by spreading the data out), LDA (class labels required) explicitly optimizes the class separation by shifting the data onto a new z-space that maximizes the ratio of between-class and within-class variance, which subsequently makes the classes more linearly separable. In our case of genre classification, this essentially meant that we are able to reduce the dimensions of our data (and therefore the risk of overfitting) whilst ensuring that we still captured the essence of what separated one genre from another. 

To apply LDA to our data, we used the ‘LinearDiscriminantAnalysis’ module from the ‘sklearn.discriminant_analysis package’. Unlike PCA, where the number of principal components generated was obtained by stating the proportion of variance we would like our new z-space to capture, LDA requires us to provide it with the number of linear discriminants. Since the maximum number of linear discriminants we can choose is n - 1, where n is the number of classes (in this case 10 genres), we chose to reduce our data to a maximum possible 9 linear discriminants. It should be noted that LDA makes three very significant assumptions about the nature of our data: features are normally distributed, features are statistically independent from one another, and the covariance matrix of our features is constant across classes (homoscedasticity). While it is rare that a data set conforms to all of these assumptions, LDA works very well in practice, as seen through the fact that both our SVM and RFC models achieved their highest accuracies when fed with the LDA formatted version of our data. 

### ML Algorithm/Model 1: Support Vector Machine (SVM)

The first classification algorithm that we decided to use is a Support Vector Machine (SVM), due to its many advantages:
1. SVM inherently finds the hyperplane that maximizes the margin between different classes, which is especially useful in music classification as some genres only have subtle differences separating them.
2. The kernel option provided by the SVM algorithm allows us to tune the model to best fit the characteristics of our data. Through alternating from the default linear kernel to more complicated options such as polynomial or radial basis function (RBF), we can shift our data to a transformed feature space without explicitly needing to compute the coordinates of the new space. Essentially, this allows us to capture non-linear relationships between our audio features, which was evident in our case as the RBF kernel yielded the highest cross-validation accuracy.
3. SVM is very well-suited to high-dimensional spaces. While we were able to reduce the number of dimensions down to 34 principal components with PCA and 9 linear discriminants with LDA, being able to use an algorithm that performs relatively well with all 57 original features is advantageous, as it would allow us to maintain some model interpretability (i.e. know which features are more important than others in genre classification).
4. While SVM is inherently a binary classifier, it can be used for multi-class classification tasks by using the one-vs-all or one-vs-one extension methods, which is required in our case since we have 10 classes.

### ML Algorithm/Model 2: Random Forest Classifier (RFC)

The second classification algorithm that we decided to use is Random Forest Classifier (RFC), due to its many advantages:
1. Because of the relatively simple structure of Random Forests (a collection of decision trees), it is easy to identify important audio features when it comes to genre classification. The decision tree layout actively quantifies how effective certain features are in the prediction process.
2. RFC is great at dealing with non-linear relationships because it is an ensemble method (RFC merges results from multiple decision trees). Due to the nature of audio data, we didn’t expect our features to be linear, so random forest was a clear choice. 
3. Again, because RFC is an ensemble method, it is far less likely to overfit the data as compared to a traditional decision tree algorithm. Since we are looking at the aggregation of several decision trees, it is much harder to overfit to the training data (a computed average is much more representative of a dataset than individual point).
4. If a RFC is able to learn your data and perform relatively well, it indicates that the data itself is learnable, and thereby encourages the pursuit of deep learning methods.



## Results and Discussion
With our project being based around classification problem with 10 classes, the following metrics will be used (with the goal values):
- Overall accuracy/Cross-validation accuracy (> 70%)
- Precision (> 0.7) (Per-class/averaged)
- Recall (> 0.7)  (Per-class/averaged)
- F1 Score (> 0.7)  (Per-class/averaged)

While we hope that we can achieve 70% and above in these metrics (which are cited as good metric levels for 10 classes), we expect that some classes will have lower metric levels (particularly similar genres that will get misidentified for one another, i.e. pop and hip-hop). This is why we will also use the 10x10 confusion matrix to detect these specific misidentifications.

### Results

We apply both models, SVM and RFC, across the three different data preprocessing types to see which would be most effective for our task. We find the following:

| <u>Avg. Accuracy | Original (no dimension reduction) | PCA    | LDA    |
|-------------------|-----------------------------------|--------|--------|
| **SVM**               | 60.523%                            | 58.219% | <u>70.117% |
| **RFC**               | 58.027 %                           | 56.168% | <u>69.351% |
*<u>Table 1</u>: Average accuracy across models / pre-processing techniques*

We observe that the two highest performing models were SVM with LDA and RFC with LDA. We then look at the accuracy reports and confusion matrices for these two models.

### Results Algorithm/Model 1: SVM with LDA

| <u>Genre  | <u>Precision | <u>Recall | <u>F1-Score |
|------------|-----------|--------|----------|
| **Blues**  | 68.138%   | 61.071%| 64.197%  |
| **Classical** | 92.185% | 93.810%| 92.401%  |
| **Country**| 63.751%   | 68.690%| 65.580%  |
| **Disco**  | 61.338%   | 65.155%| 62.431%  |
| **HipHop** | 66.044%   | 66.024%| 65.035%  |
| **Jazz**   | 82.460%   | 85.702%| 83.714%  |
| **Metal**  | 78.184%   | 78.333%| 77.768%  |
| **Pop**    | 81.880%   | 76.690%| 78.805%  |
| **Reggae** | 61.923%   | 62.702%| 61.375%  |
| **Rock**   | 47.108%   | 42.988%| 43.913%  |
*<u>Table 2</u>: Accuracy report for SVM with LDA*

![svm_lda](./images/svm_lda.png)
*<u>Table 3</u>: Confusion matrix of SVM with LDA*

### Results Algorithm/Model 2: RFC with LDA

| <u>Genre   | <u>Precision | <u>Recall  | <u>F1-Score |
|-------------|-----------|---------|----------|
| **Blues**   | 66.723%   | 63.417% | 64.926%  |
| **Classical** | 93.898% | 90.476% | 91.238%  |
| **Country** | 64.734%   | 70.393% | 66.969%  |
| **Disco**   | 59.506%   | 61.833% | 60.140%  |
| **HipHop**  | 64.892%   | 61.619% | 62.447%  |
| **Jazz**    | 78.692%   | 84.821% | 81.103%  |
| **Metal**   | 79.770%   | 81.726% | 80.303%  |
| **Pop**     | 81.868%   | 77.440% | 79.000%  |
| **Reggae**  | 57.213%   | 60.762% | 58.133%  |
| **Rock**    | 49.074%   | 41.024% | 43.886%  |
*<u>Table 4/u>: Accuracy report for RFC with LDA*

![rfc_lda](./images/rfc_lda.png)
*<u>Table 5</u>: Confusion matrix of RFC with LDA*

### Discussion of Results

When looking at the average accuracy of the models across the three different data pre-processing techniques (Table 1), we find two key observations:
1. Our first observation is that PCA worsened both our models and LDA improved both our models. We initially believed this had to do with our threshold for PCA. We set PCA to have a threshold of capturing 90% of the variance in the data (with 23 components), but after changing it to 95% (with 34 components), we obtained the results above. The results above are better than the original results, but still maintain the observation that PCA worsened performance on both models, suggesting that PCA is not the right pre-processing technique for this dataset.

    There are several possible reasons for this, but we suspect it has to do with reducing data with respect to separability. PCA tries to reduce dimensionality by finding principal components that maximize variance in the dataset. This means that PCA tries to reduce the amount of data while keeping it as representative as possible, but doesn’t necessarily help with class separability. It may be that the variance in our data doesn’t tend to lead to distinctions, so PCA would reduce the dataset to dimensions that makes classes seem more similar, leading to worsened performance. On the other hand, LDA tries to reduce dimensionality while maintaining class separability. By maximizing the ratio of between-class variance to within-class variance, LDA ensures that the reduced dimensions are effective for class separation. Because our task is classification and we have labeled training data, it makes sense that we use LDA and that it performs better, as its purpose aligns better with our goals.
2. Our second observation is that SVM performed better than RFC for all pre-processing techniques. We believe that this is due to the characteristics of our data. Both SVM and RFC work well with non-linear data; however, SVM works better when the data and therefore the decisions are more complex.

The confusion matrices above highlight specifically which genres are easy or difficult to label. As you can see, Classical, Jazz, and Metal all have over an 80% accuracy in predicting their respective genres for both SVM and RFC with LDA. You can also see from both models rock was by far the hardest genre to identify (43% and 41%). Through the SVM/LDA combination, we have met our original quantitative goal of achieving an average cross-validation accuracy of 70% and above, but the confusion matrix reinforces our original hypothesis that certain genres were going to be more difficult to differentiate than others.

### Next Steps

Thus far, we have shown the application of non-deep learning methods for music genre classification. Specifically, through the use of LDA with SVM and RFC which yielded promising results with accuracies as high as 70%. The scores from these models demonstrate the learnability of our dataset and their effectiveness in classifying music genres. Moving forward, we want to explore the findings of the cited paper “Deep learning for audio-based music classification and tagging,” to explore deep learning models such as ANN and RNN to further increase our classification accuracy. 
For results, we want to include more visualizations on  avg. accuracy over folds for SVM/RFC and loss over epochs for ANN/RNN in our final paper. This would allow us to have more insight into the training process as opposed to just looking at the final results. With these visualizations, we can explore the deviation between training and validation to evaluate performance, and whether the model overfits or underfits.

## References
[1] M. Chaudhury, A. Karami, and M. A. Ghazanfar, “Large-scale music genre analysis and classification using Machine Learning with apache spark,” Electronics, vol. 11, no. 16, p. 2567, Aug. 2022. doi:10.3390/electronics11162567 

[2] J. Nam, K. Choi, J. Lee, S.-Y. Chou, and Y.-H. Yang, “Deep learning for audio-based music classification and tagging: Teaching computers to distinguish rock from Bach,” IEEE Signal Processing Magazine, vol. 36, no. 1, pp. 41–51, Dec. 2018. doi:10.1109/msp.2018.2874383 

[3] H. Purwins et al., “Deep Learning for Audio Signal Processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 13, no. 2, pp. 206–219, May 2019. doi:10.1109/jstsp.2019.2908700 

## Gantt Chart
![gantt](./images/gantt.png)

## Contribution Table
| Name  | Midterm contributions |
|-------|------------------------|
| Ariya | RFC implementation, evaluation |
| Akaash| Results, visualizations, discussion |
| Dorsa | Data Preprocessing (PCA) |
| Nikan | Data Preprocessing (LDA), Data Preprocessing (Audio Augmentation) |
| Stella| SVM implementation, evaluation |


