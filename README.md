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

Zbiór treningowy składa się z 374 próbek, z których każda ma 1808 cech, oraz odpowiadających im 374 etykiet. Zbiór walidacyjny zawiera 80 próbek, również z 1808 cechami każda, i odpowiadających im 80 etykiet. Zbiór testowy składa się z 81 próbek, każda z 1808 cechami, oraz 81 etykiet. Zestaw treningowy jest największy, co jest typowe, ponieważ służy do trenowania modelu, natomiast zestaw walidacyjny i testowy są mniejsze i służą odpowiednio do tuningu hiperparametrów oraz oceny ostatecznej wydajności modelu.
