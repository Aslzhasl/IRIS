# Iris Recognition

This project implements an iris recognition pipeline in Python. It loads iris
images from person/class folders, preprocesses them as grayscale 128x128 images,
extracts classical computer vision features, trains multiple machine learning
models, trains a PyTorch CNN, and compares the final results.

## Dataset Structure

Place the dataset in `data/raw/iris`. Each subfolder is treated as one
class/person.

```text
data/raw/iris/
  person_name_1/
    img_0001.jpg
    img_0002.jpg
  person_name_2/
    img_0001.jpg
```

## Algorithms Used

- LBP features + SVM
- HOG features + RandomForest
- ORB features + KNN
- Convolutional Neural Network (CNN)
- Soft Voting Ensemble using SVM, RandomForest, and KNN

## Install Dependencies on Windows

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Project

Make sure the dataset exists at:

```text
data/raw/iris
```

Run the full pipeline:

```powershell
python main.py
```

The script saves:

- Trained models in `models/`
- Metrics CSV in `results/results.csv`
- Confusion matrix plots in `results/`

## Results

| Model | Accuracy | Precision Macro | Recall Macro | F1 Macro |
| --- | ---: | ---: | ---: | ---: |
| SVM_LBP | 0.558594 | 0.564149 | 0.558594 | 0.508343 |
| Voting_Ensemble_LBP | 0.777344 | 0.777073 | 0.777344 | 0.774354 |
| RandomForest_HOG | 0.972656 | 0.976649 | 0.972656 | 0.973242 |
| KNN_ORB | 0.367188 | 0.392642 | 0.367188 | 0.313608 |
| CNN | 0.980469 | 0.982001 | 0.980469 | 0.980664 |

## Conclusion

The CNN achieved the best overall performance, followed closely by the
HOG + RandomForest model. LBP features improved significantly when used with
the voting ensemble, while ORB + KNN performed the weakest on this dataset.
For this iris recognition task, learned CNN features and HOG descriptors are
the strongest approaches among the tested methods.
