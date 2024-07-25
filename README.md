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

link: [https://www.kaggle.com/datasets/ejlok1/cremad/data](https://www.kaggle.com/datasets/nilanshk/emotion-classification-speach/data)

The RAVDESS dataset, a comprehensive collection designed for emotional speech and song analysis, consists of 7356 unique files. Each file is meticulously named with a 7-part numerical identifier that encapsulates various stimulus characteristics, facilitating detailed analysis and research. Below is a detailed explanation of the filename convention and its components.
Filename Convention

Each RAVDESS file is named using a specific format that includes the following seven numerical identifiers, structured as 02-01-06-01-02-01-12.mp4:

    Emotion:
        01: Neutral
        02: Calm
        03: Happy
        04: Sad
        05: Angry
        06: Fearful
        07: Disgust
        08: Surprised

        
### Feature Extraction

Feature extraction from an audio signal involves several key steps. First, pre-emphasis is applied to amplify higher frequencies, followed by dividing the signal into short overlapping frames using a Hamming window to minimize spectral leakage. Next, the Fourier transform is used to move to the frequency domain and calculate the power spectrum. Important features such as Mel-Frequency Cepstral Coefficients (MFCC), Zero-Crossing Rate (ZCR), short-term energy, pitch, and formants (using Linear Predictive Coding) are then extracted. Statistics like minimum, maximum, mean, and variance are calculated for each feature, and all features are combined into a single feature vector for further analysis and emotion classification in machine learning models.

### description of data division no. 1

The training set consists of 374 samples, each with 1808 features, and their corresponding 374 labels. The validation set contains 80 samples, also with 1808 features each, and their corresponding 80 labels. The test set consists of 81 samples, each with 1808 features, and 81 labels. The training set is the largest, which is typical as it is used for training the model, while the validation and test sets are smaller and are used for hyperparameter tuning and final model performance evaluation, respectively.

### description of data division no. 2

The training set consists of 1008 samples, each with 20 features, and their corresponding 1008 labels. The validation set contains 216 samples, also with 20 features each, and their corresponding 216 labels. The test set consists of 216 samples, each with 20 features, and 216 labels. The training set is the largest, which is typical as it is used for training the model, while the validation and test sets are smaller and are used for hyperparameter tuning and final model performance evaluation, respectively.

### Steps to Reproduce

    Download the Repository:
        Clone or download the repository to your local machine.

    Update Directory Path in train_dbn.py:
        Open the train_dbn.py file.
        Update the variable directory to your local path where data1 is stored.

    Run train_dbn.py:
        Execute the train_dbn.py file to start the training process.

    Update Directory Path in train_svm.py:
        Open the train_svm.py file.
        Update the variable directory to your local path where the speech file is located.

    Run train_svm.py:
        Execute the train_svm.py file to train the SVM model.

    Run train_svm_dbn.py:
        Execute the train_svm_dbn.py file to complete the final training step.

By following these steps, you should be able to reproduce the results of the project. Make sure the directory paths are correctly set to avoid any errors during execution.


# RESULTS

DBN on  data no. 1 :

| WYMIARY SIECI          | EPOCHS + BATCH SIZE + LEARNING RATE                     | BŁĄD                      |
|------------------------|---------------------------------------------------------|---------------------------|
| 400, 200, 150, 100     | epochs=100, batch_size=32, learning rate=0,01           | 1.0127983093261719        |
| 128, 64, 32            | epochs=100, batch_size=32, learning rate=0,01           | 1.0081210136413574        |
| 128, 64, 32, 16, 8     | epochs=100, batch_size=32, learning rate=0,01           | 1.0088634490966797        |
| 10, 100                | epochs=100, batch_size=32, learning rate=0,01           | 1.00207173824310          |
| 10, 50, 100            | epochs=10, batch_size=32, learning rate=0,1             | 1.0008162260055542        |
| 10, 50, 100            | epochs=10, batch_size=32, learning rate=0,001           | 0.06289192289113998       |
| 128, 64, 32, 16        | epochs=10, batch_size=32, learning rate=0,01            | 1.0014728307724           |
| 100, 5, 1              | epochs=100, batch_size=128, learning rate=0,01          | 1.0137122869491577        |
| 128, 64, 32            | epochs=100, batch_size=128, learning rate=0,001         | 1.0054868459701538        |
| 10, 50, 100            | epochs=10, batch_size=32, learning rate=0,01 + normalizacja | 0.2866422253965729        |
| 128, 64, 32            | epochs=100, batch_size=128, learning rate=0,001+ normalizacja | 1.0053935050964355      |



DBN on  data no. 2 :

| WYMIARY SIECI          | EPOCHS + BATCH SIZE + LEARNING RATE                     | BŁĄD                      |
|------------------------|---------------------------------------------------------|---------------------------|
| 400, 200, 150, 100     | epochs=100, batch_size=32, learning rate=0,01           | 1.0013967752456665       
| 128, 64, 32     | epochs=100, batch_size=32, learning rate=0,01           | 1.0013967752456665       
| 128, 64, 32, 16, 8     | epochs=100, batch_size=32, learning rate=0,01           | 1.0191926956176758        |
| 10, 100                | epochs=100, batch_size=128, learning rate=0,01           | 1.000476598739624         |
| 10, 50, 100            | epochs=10, batch_size=128, learning rate=0,1             | 1.0007373094558716        |
| 10, 50, 100            | epochs=10, batch_size=128, learning rate=0,001           | 1.0004500150680542      |
| 128, 64, 32, 16        | epochs=10, batch_size=32, learning rate=0,1            | 1.00066244602203372          |
| 100, 5, 1              | epochs=100, batch_size=128, learning rate=0,01          | 1.0137122869491577        |
| 128, 64, 32            | epochs=100, batch_size=128, learning rate=0,001         | 1.0054868459701538        |
| 10, 50, 100            | epochs=10, batch_size=128, learning rate=0,01 + normalizacja | 1.0004647970199585       |
| 128, 64, 32            | epochs=100, batch_size=128, learning rate=0,001+ normalizacja | 1.0053935050964355      |


SVM on  data no. 1:


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

SVM on  data no. 1:

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


SVM on  data no. 2 

'num_centers': [50, 75, 100, 125, 150, 175, 200],
    'beta': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]}

| num_centers | beta | Validation Accuracy |
|-------------|------|---------------------|
| 50          | 0.25 | 6.94%               |
| 50          | 0.5  | 9.72%               |
| 50          | 0.75 | 5.56%               |
| 50          | 1.0  | 6.94%               |
| 50          | 1.25 | 5.56%               |
| 50          | 1.5  | 5.56%               |
| 50          | 2.0  | 15.28%              |
| 75          | 0.25 | 4.17%               |
| 75          | 0.5  | 11.11%              |
| 75          | 0.75 | 8.33%               |
| 75          | 1.0  | 5.56%               |
| 75          | 1.25 | 5.56%               |
| 75          | 1.5  | 5.56%               |
| 75          | 2.0  | 1.39%               |
| 100         | 0.25 | 5.56%               |
| 100         | 0.5  | 6.94%               |
| 100         | 0.75 | 9.72%               |
| 100         | 1.0  | 6.94%               |
| 100         | 1.25 | 8.33%               |
| 100         | 1.5  | 11.11%              |
| 100         | 2.0  | 6.94%               |
| 125         | 0.25 | 8.33%               |
| 125         | 0.5  | 15.28%              |
| 125         | 0.75 | 11.11%              |
| 125         | 1.0  | 9.72%               |
| 125         | 1.25 | 8.33%               |
| 125         | 1.5  | 8.33%               |
| 125         | 2.0  | 12.50%              |
| 150         | 0.25 | 9.72%               |
| 150         | 0.5  | 9.72%               |
| 150         | 0.75 | 12.50%              |
| 150         | 1.0  | 5.56%               |
| 150         | 1.25 | 6.94%               |
| 150         | 1.5  | 8.33%               |
| 150         | 2.0  | 8.33%               |
| 175         | 0.25 | 9.72%               |
| 175         | 0.5  | 6.94%               |
| 175         | 0.75 | 19.44%              |
| 175         | 1.0  | 13.89%              |
| 175         | 1.25 | 11.11%              |
| 175         | 1.5  | 5.56%               |
| 175         | 2.0  | 15.28%              |
| 200         | 0.25 | 8.33%               |
| 200         | 0.5  | 6.94%               |
| 200         | 0.75 | 11.11%              |
| 200         | 1.0  | 9.72%               |
| 200         | 1.25 | 8.33%               |
| 200         | 1.5  | 13.89%              |
| 200         | 2.0  | 11.11%              |

**Best Parameters:** {'num_centers': 175, 'beta': 0.75}  
**Best Validation Accuracy:** 19.44%  
**Test Accuracy with best model:** 13.89%

SVM on  data no. 2 

'num_centers': [8, 12, 16, 20, 24, 28, 32, 36],
    'beta': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

| num_centers | beta | Validation Accuracy |
|-------------|------|---------------------|
| 8           | 0.25 | 13.43%              |
| 8           | 0.5  | 10.19%              |
| 8           | 0.75 | 12.50%              |
| 8           | 1.0  | 14.35%              |
| 8           | 1.25 | 8.33%               |
| 8           | 1.5  | 16.20%              |
| 8           | 2.0  | 12.04%              |
| 12          | 0.25 | 13.43%              |
| 12          | 0.5  | 12.50%              |
| 12          | 0.75 | 11.57%              |
| 12          | 1.0  | 16.20%              |
| 12          | 1.25 | 14.35%              |
| 12          | 1.5  | 11.57%              |
| 12          | 2.0  | 13.43%              |
| 16          | 0.25 | 13.43%              |
| 16          | 0.5  | 14.81%              |
| 16          | 0.75 | 12.04%              |
| 16          | 1.0  | 16.20%              |
| 16          | 1.25 | 12.04%              |
| 16          | 1.5  | 16.67%              |
| 16          | 2.0  | 10.19%              |
| 20          | 0.25 | 12.50%              |
| 20          | 0.5  | 10.19%              |
| 20          | 0.75 | 13.43%              |
| 20          | 1.0  | 13.43%              |
| 20          | 1.25 | 14.81%              |
| 20          | 1.5  | 16.20%              |
| 20          | 2.0  | 10.19%              |
| 24          | 0.25 | 12.96%              |
| 24          | 0.5  | 16.20%              |
| 24          | 0.75 | 10.65%              |
| 24          | 1.0  | 9.72%               |
| 24          | 1.25 | 11.57%              |
| 24          | 1.5  | 12.50%              |
| 24          | 2.0  | 15.28%              |
| 28          | 0.25 | 12.04%              |
| 28          | 0.5  | 12.04%              |
| 28          | 0.75 | 10.65%              |
| 28          | 1.0  | 15.74%              |
| 28          | 1.25 | 11.57%              |
| 28          | 1.5  | 12.04%              |
| 28          | 2.0  | 11.11%              |
| 32          | 0.25 | 12.96%              |
| 32          | 0.5  | 15.74%              |
| 32          | 0.75 | 9.26%               |
| 32          | 1.0  | 15.74%              |
| 32          | 1.25 | 12.96%              |
| 32          | 1.5  | 9.26%               |
| 32          | 2.0  | 13.43%              |
| 36          | 0.25 | 12.50%              |
| 36          | 0.5  | 12.04%              |
| 36          | 0.75 | 12.96%              |
| 36          | 1.0  | 14.35%              |
| 36          | 1.25 | 15.74%              |
| 36          | 1.5  | 16.20%              |
| 36          | 2.0  | 17.13%              |

**Best Parameters:** {'num_centers': 36, 'beta': 2.0}  
**Best Validation Accuracy:** 17.13%  
**Test Accuracy with best model:** 13.89%


SVM+DBN 


| num_centers | beta | Validation Accuracy |
|-------------|------|---------------------|
| 8           | 0.25 | 20.00%              |
| 8           | 0.5  | 20.00%              |
| 8           | 0.75 | 20.00%              |
| 8           | 1.0  | 21.25%              |
| 8           | 1.25 | 20.00%              |
| 8           | 1.5  | 17.50%              |
| 8           | 2.0  | 21.25%              |
| 12          | 0.25 | 20.00%              |
| 12          | 0.5  | 20.00%              |
| 12          | 0.75 | 20.00%              |
| 12          | 1.0  | 22.50%              |
| 12          | 1.25 | 15.00%              |
| 12          | 1.5  | 16.25%              |
| 12          | 2.0  | 20.00%              |
| 16          | 0.25 | 20.00%              |
| 16          | 0.5  | 20.00%              |
| 16          | 0.75 | 20.00%              |
| 16          | 1.0  | 16.25%              |
| 16          | 1.25 | 21.25%              |
| 16          | 1.5  | 20.00%              |
| 16          | 2.0  | 21.25%              |
| 20          | 0.25 | 20.00%              |
| 20          | 0.5  | 20.00%              |
| 20          | 0.75 | 20.00%              |
| 20          | 1.0  | 15.00%              |
| 20          | 1.25 | 26.25%              |
| 20          | 1.5  | 17.50%              |
| 20          | 2.0  | 13.75%              |
| 24          | 0.25 | 20.00%              |
| 24          | 0.5  | 20.00%              |
| 24          | 0.75 | 20.00%              |
| 24          | 1.0  | 22.50%              |
| 24          | 1.25 | 20.00%              |
| 24          | 1.5  | 15.00%              |
| 24          | 2.0  | 18.75%              |
| 28          | 0.25 | 20.00%              |
| 28          | 0.5  | 20.00%              |
| 28          | 0.75 | 20.00%              |
| 28          | 1.0  | 23.75%              |
| 28          | 1.25 | 18.75%              |
| 28          | 1.5  | 18.75%              |
| 28          | 2.0  | 17.50%              |
| 32          | 0.25 | 20.00%              |
| 32          | 0.5  | 20.00%              |
| 32          | 0.75 | 20.00%              |
| 32          | 1.0  | 21.25%              |
| 32          | 1.25 | 15.00%              |
| 32          | 1.5  | 17.50%              |
| 32          | 2.0  | 15.00%              |
| 36          | 0.25 | 20.00%              |
| 36          | 0.5  | 20.00%              |
| 36          | 0.75 | 20.00%              |
| 36          | 1.0  | 20.00%              |
| 36          | 1.25 | 12.50%              |
| 36          | 1.5  | 13.75%              |
| 36          | 2.0  | 12.50%              |

**Best Parameters:** {'num_centers': 20, 'beta': 1.25}  
**Best Validation Accuracy:** 26.25%  
**Validation Accuracy of DBN with RBF:** 20.00%

### Model Parameters

| Layer | W Shape              | bp Shape           | bn Shape         | stddev         |
|-------|----------------------|--------------------|------------------|----------------|
| 0     | torch.Size([143652, 64]) | torch.Size([64])   | torch.Size([143652]) | torch.Size([]) |
| 1     | torch.Size([64, 32])     | torch.Size([32])   | torch.Size([64])     | None           |
| 2     | torch.Size([32, 16])     | torch.Size([16])   | torch.Size([32])     | None           |


