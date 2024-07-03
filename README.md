# tabular-regression
## Overview
The goal of this project is to build a regression model using a given dataset. It includes exploratory data analysis and scripts for model training and making predictions on the test data.

## Project Structure

```
tabular-regression/
│
├── notebooks/
│ └── eda.ipynb
│
├── scripts/
│ ├── train.py
│ ├── predict.py
│
├── data/
│ ├── train.csv
│ ├── hidden_test.csv
│ └── predictions.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```
Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```
Create a virtual environment and activate it:
```bash
python -m venv env
source env/bin/activate
```
Install the required packages:
```bash
pip install -r requirements.txt
```
Train the model:
```bash
python scripts/train.py --train_path data/train.csv --model_path model.pkl --scaler_path scaler.pkl
```
Make predictions on the test data:
```bash
python scripts/predict.py --test_path data/hidden_test.csv --model_path model.pkl --scaler_path scaler.pkl --output_path data/predictions.csv
```
