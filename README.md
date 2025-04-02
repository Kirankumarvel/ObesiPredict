
# ObesiPredict
ObesiPredict 🎯- A sleek and simple name that combines "Obesity" and "Predict."

ObesiPredict is a machine learning pipeline for predicting obesity levels based on various health and lifestyle factors. It utilizes logistic regression models with One-vs-All (OvA) and One-vs-One (OvO) strategies to classify individuals into different obesity categories.

## Project Structure

```
ObesiPredict/
│── data/
│   ├── Obesity_level_prediction_dataset.csv  # Dataset (if stored locally)
│── notebooks/
│   ├── Exploratory_Data_Analysis.ipynb  # Initial data exploration
│── src/
│   ├── preprocess.py  # Data preprocessing functions
│   ├── train.py  # Model training and evaluation
│── ObesiPredict.py  # Main script for the obesity prediction pipeline
│── requirements.txt  # Dependencies
│── README.md  # Project documentation
```

## Installation

Clone the repository:
```bash
git clone https://github.com/kirankumarvel/ObesiPredict.git
cd ObesiPredict
```

Create a virtual environment (optional but recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset used is publicly available at:
[Obesity-level prediction dataset](https://github.com/Kirankumarvel/ObesiPredict/blob/main/Obesity_level_prediction_dataset.csv)

## Running the Pipeline

To execute the full pipeline:
```bash
python ObesiPredict.py
```

## Features Implemented

- **Exploratory Data Analysis (EDA)**
- **Data Preprocessing:** Standardization, one-hot encoding
- **Model Training:** Logistic Regression with OvA & OvO
- **Performance Evaluation:** Accuracy, classification report
- **Feature Importance Visualization**

## Next Steps

- Try different classification models
- Experiment with hyperparameter tuning
- Add support for real-time predictions via a web API

## License

This project is open-source under the MIT License.
