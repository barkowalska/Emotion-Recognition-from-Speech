# Emotion-Recognition-from-Speech

The project involves evaluating three models: an SVM model, a DBN model, and an SVM model combined with a DBN.




###### Article
Emotion Recognition from Chinese Speech for Smart Affective Services Using a Combination of SVM and DBN

link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5539696/

Authors: Lianzhang Zhu, Leiming Chen, Dehai Zhao, Jiehan Zhou and Weishan Zhang1

### Description of the found data no. 1:

link: https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb?resource=download

The EMODB dataset, developed by the Institute of Communication Science at the Technical University of Berlin, is a publicly accessible collection of German emotional speech recordings. This dataset comprises 535 utterances from ten seasoned speakers, equally divided between male and female. The dataset captures seven distinct emotions: anger, boredom, fear, joy, sadness, disgust, and neutrality. The recordings were initially captured at a 48 kHz sampling rate and subsequently down-sampled to 16 kHz for standardization.
Naming Structure of Files

Each file in the dataset adheres to a specific naming scheme:

    Positions 1-2: Identifier for the speaker
    Positions 3-5: Code representing the text spoken
    Position 6: Letter indicating the emotion (German abbreviation)
    Position 7: Version identifier (if multiple versions exist)

### Description of the found data no. 2:

link: https://www.kaggle.com/datasets/ejlok1/cremad/data

The CREMA-D dataset, developed by a consortium of researchers, is a publicly accessible collection of American English emotional speech recordings. This dataset comprises 7,442 original clips from 91 actors, including 48 male and 43 female actors, ranging in age from 20 to 74. The actors represent a variety of races and ethnicities, including African American, Asian, Caucasian, Hispanic, and Unspecified. Each actor spoke from a selection of 12 sentences, presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad) and four different emotion levels (Low, Medium, High, and Unspecified).

Each file in the dataset adheres to a specific naming scheme:

    Actor ID: A 4-digit number at the start of the file identifies the actor.
    Sentence Code: Represents the text spoken using a three-letter acronym.
    Emotion Code: Indicates the emotion using a three-letter code.
    Emotion Level: Indicates the emotion level using a two-letter code.
    
### Feature Extraction

Feature extraction from an audio signal involves several key steps. First, pre-emphasis is applied to amplify higher frequencies, followed by dividing the signal into short overlapping frames using a Hamming window to minimize spectral leakage. Next, the Fourier transform is used to move to the frequency domain and calculate the power spectrum. Important features such as Mel-Frequency Cepstral Coefficients (MFCC), Zero-Crossing Rate (ZCR), short-term energy, pitch, and formants (using Linear Predictive Coding) are then extracted. Statistics like minimum, maximum, mean, and variance are calculated for each feature, and all features are combined into a single feature vector for further analysis and emotion classification in machine learning models.

### description of data division no. 1

The training set consists of 374 samples, each with 1808 features, and their corresponding 374 labels. The validation set contains 80 samples, also with 1808 features each, and their corresponding 80 labels. The test set consists of 81 samples, each with 1808 features, and 81 labels. The training set is the largest, which is typical as it is used for training the model, while the validation and test sets are smaller and are used for hyperparameter tuning and final model performance evaluation, respectively.

SVM:

'num_centers': [50, 75, 100, 125, 150, 175, 200],
    'beta': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

| num_centers | beta | Validation Accuracy |
|-------------|------|---------------------|
| 50          | 0.25 | 23.75%              |
| 50          | 0.5  | 23.75%              |
| 50          | 0.75 | 20.00%              |
| 50          | 1.0  | 21.25%              |
| 50          | 1.25 | 22.50%              |
| 50          | 1.5  | 17.50%              |
| 50          | 2.0  | 15.00%              |
| 75          | 0.25 | 21.25%              |
| 75          | 0.5  | 20.00%              |
| 75          | 0.75 | 18.75%              |
| 75          | 1.0  | 22.50%              |
| 75          | 1.25 | 18.75%              |
| 75          | 1.5  | 16.25%              |
| 75          | 2.0  | 17.50%              |
| 100         | 0.25 | 21.25%              |
| 100         | 0.5  | 21.25%              |
| 100         | 0.75 | 22.50%              |
| 100         | 1.0  | 21.25%              |
| 100         | 1.25 | 23.75%              |
| 100         | 1.5  | 17.50%              |
| 100         | 2.0  | 10.00%              |
| 125         | 0.25 | 23.75%              |
| 125         | 0.5  | 22.50%              |
| 125         | 0.75 | 28.75%              |
| 125         | 1.0  | 27.50%              |
| 125         | 1.25 | 26.25%              |
| 125         | 1.5  | 21.25%              |
| 125         | 2.0  | 13.75%              |
| 150         | 0.25 | 25.00%              |
| 150         | 0.5  | 23.75%              |
| 150         | 0.75 | 18.75%              |
| 150         | 1.0  | 21.25%              |
| 150         | 1.25 | 16.25%              |
| 150         | 1.5  | 20.00%              |
| 150         | 2.0  | 13.75%              |
| 175         | 0.25 | 22.50%              |
| 175         | 0.5  | 23.75%              |
| 175         | 0.75 | 21.25%              |
| 175         | 1.0  | 23.75%              |
| 175         | 1.25 | 23.75%              |
| 175         | 1.5  | 16.25%              |
| 175         | 2.0  | 20.00%              |
| 200         | 0.25 | 22.50%              |
| 200         | 0.5  | 26.25%              |
| 200         | 0.75 | 17.50%              |
| 200         | 1.0  | 25.00%              |
| 200         | 1.25 | 16.25%              |
| 200         | 1.5  | 18.75%              |
| 200         | 2.0  | 23.75%              |

**Best Parameters:** {'num_centers': 125, 'beta': 0.75}  
**Best Validation Accuracy:** 28.75%  
**Test Accuracy with best model:** 18.52%


'num_centers': [8, 12, 16, 20, 24, 28, 32, 36],
    'beta': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    | num_centers | beta | Validation Accuracy |
|-------------|------|---------------------|
| 8           | 0.25 | 23.75%              |
| 8           | 0.5  | 31.25%              |
| 8           | 0.75 | 15.00%              |
| 8           | 1.0  | 15.00%              |
| 8           | 1.25 | 17.50%              |
| 8           | 1.5  | 13.75%              |
| 8           | 2.0  | 15.00%              |
| 12          | 0.25 | 21.25%              |
| 12          | 0.5  | 25.00%              |
| 12          | 0.75 | 11.25%              |
| 12          | 1.0  | 25.00%              |
| 12          | 1.25 | 13.75%              |
| 12          | 1.5  | 15.00%              |
| 12          | 2.0  | 25.00%              |
| 16          | 0.25 | 23.75%              |
| 16          | 0.5  | 22.50%              |
| 16          | 0.75 | 22.50%              |
| 16          | 1.0  | 27.50%              |
| 16          | 1.25 | 13.75%              |
| 16          | 1.5  | 23.75%              |
| 16          | 2.0  | 20.00%              |
| 20          | 0.25 | 20.00%              |
| 20          | 0.5  | 21.25%              |
| 20          | 0.75 | 21.25%              |
| 20          | 1.0  | 11.25%              |
| 20          | 1.25 | 23.75%              |
| 20          | 1.5  | 12.50%              |
| 20          | 2.0  | 26.25%              |
| 24          | 0.25 | 22.50%              |
| 24          | 0.5  | 28.75%              |
| 24          | 0.75 | 20.00%              |
| 24          | 1.0  | 20.00%              |
| 24          | 1.25 | 16.25%              |
| 24          | 1.5  | 18.75%              |
| 24          | 2.0  | 17.50%              |
| 28          | 0.25 | 22.50%              |
| 28          | 0.5  | 22.50%              |
| 28          | 0.75 | 20.00%              |
| 28          | 1.0  | 11.25%              |
| 28          | 1.25 | 13.75%              |
| 28          | 1.5  | 23.75%              |
| 28          | 2.0  | 15.00%              |
| 32          | 0.25 | 28.75%              |
| 32          | 0.5  | 18.75%              |
| 32          | 0.75 | 15.00%              |
| 32          | 1.0  | 13.75%              |
| 32          | 1.25 | 20.00%              |
| 32          | 1.5  | 12.50%              |
| 32          | 2.0  | 13.75%              |
| 36          | 0.25 | 21.25%              |
| 36          | 0.5  | 23.75%              |
| 36          | 0.75 | 20.00%              |
| 36          | 1.0  | 22.50%              |
| 36          | 1.25 | 17.50%              |
| 36          | 1.5  | 13.75%              |
| 36          | 2.0  | 16.25%              |

**Best Parameters:** {'num_centers': 8, 'beta': 0.5}  
**Best Validation Accuracy:** 31.25%  
**Test Accuracy with best model:** 16.05%
