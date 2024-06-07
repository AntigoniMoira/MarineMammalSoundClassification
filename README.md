
# Marine Mammal Sound Classification

The purpose of this project is to analyze and classify marine mammal sounds.

## Dataset

The dataset selected is the **Best of Watkins Marine Mammal Sound Database**[^1]. This dataset contains 1,694 sound cuts deemed to be of higher sound quality and lower noise from 32 different species.

![](./images/species_1_del.png)
![](./images/species_2_del.png)
![](./images/species_3_del.png)

*Species marked with an :x: were excluded from the study either because they had an extremely small number of instances (<20) or because there was a problem with the format of their `.wav` files.*

---
## Implementation Plan

The implementation plan consists of the following steps:

1. Collection of data and metadata [`scraper.ipynb`](https://github.com/AntigoniMoira/MarineMammalSoundClassification/blob/main/scraper.ipynb)
2. Exploratory data analysis [`EDA-v1.ipynb`](https://github.com/AntigoniMoira/MarineMammalSoundClassification/blob/main/EDA-v1.ipynb)
3. Data Preprocessing (Cleaning and Splitting) [`data_preprocessing.ipynb`](https://github.com/AntigoniMoira/MarineMammalSoundClassification/blob/main/data_preprocessing.ipynb)
4. Handcrafted Features and SVM as baseline for comparison [`baseline_svm.ipynb`](https://github.com/AntigoniMoira/MarineMammalSoundClassification/blob/main/baseline_svm.ipynb)
5. Handcrafted Features and Fully Connected Neural Network [`FullyConnected_NN.ipynb`](https://github.com/AntigoniMoira/MarineMammalSoundClassification/blob/main/FullyConnected_NN.ipynb)
6. Melgrams/Spectrograms and CNNs
7. RNNs/LSTMs
8. Audio Transformer
9. Transfer learning

---

[^1]: [Best of Watkins Marine Mammal Sound Database](https://whoicf2.whoi.edu/science/B/whalesounds/index.cfm)