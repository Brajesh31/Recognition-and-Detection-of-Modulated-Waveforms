# Modulation Recognition and Detection System  

## 📌 Overview  
This project focuses on **automatic recognition and classification of modulated waveforms** using **machine learning**. It generates, processes, and classifies signals like **QPSK, 8PSK, 16APSK, 32APSK, and 64APSK** using a **RandomForestClassifier** with feature extraction and visualization.  

## 🚀 Features  
✔ **Automatic Modulation Recognition** – Detects modulation type from input signals.  
✔ **Machine Learning Model** – Uses **RandomForestClassifier** to classify waveforms.  
✔ **Feature Extraction** – Extracts **statistical** and **frequency-domain** features from signals.  
✔ **Signal Visualization** – Includes **constellation diagrams, scatter plots, and confusion matrices**.  
✔ **Interactive CLI** – Allows users to **train models, test signals, and view results**.  

## 🛠️ Technologies Used  
- **Python**  
- **NumPy** – Signal processing and numerical computations.  
- **Matplotlib & Seaborn** – Data visualization.  
- **Scikit-learn** – Machine learning model training.  
- **Joblib** – Model saving and loading.  

## 📂 Project Structure  
Modulation_Recognition/
├── 📄 modulation_signal_generator.py # Generates modulation signals and saves them to files
├── 📄 modulation_recognition.py # Trains ML model & classifies signals
├── 📂 sample_inputs/ # Stores generated signals
├── 📄 modulation_model.pkl # Trained model file
├── 📄 README.md # Documentation


## 📌 Project Description  

This project focuses on **Automatic Modulation Recognition (AMR)** for **DVB-S2X waveforms** using **machine learning**.  
It includes signal generation, training a classification model, and recognizing modulation types.

## 📜 Files and Directories  

- **modulation_signal_generator.py** – Generates various modulation signals and stores them as files in `sample_inputs/`.  
- **modulation_recognition.py** – Trains a machine learning model for modulation recognition and classifies signals.  
- **sample_inputs/** – Stores generated modulation signal files.  
- **modulation_model.pkl** – The trained machine learning model for modulation classification.  
- **README.md** – Project documentation.

## 🚀 How to Run  

1. **Generate modulation signals**  
   python modulation_signal_generator.py

## 🔧 Installation  
git clone https://github.com/yourusername/Modulation_Recognition.git
cd Modulation_Recognition
pip install -r requirements.txt


## 🎯 Usage
1. Generate Modulation Signals
bash
Copy
Edit
python modulation_signal_generator.py
👉 Saves QPSK, 8PSK, 16APSK, 32APSK, and 64APSK signals as text files.

2. Train & Test the Model
bash
Copy
Edit
python modulation_recognition.py
Option 1: Train a new model with custom sample size.
Option 2: Test a new waveform signal file and get predictions.
## 📊 Model Performance
The trained RandomForestClassifier achieves an accuracy of 80% on test data.

## 📜 License
This project is open-source and licensed under the MIT License.

Last updated: 2026-10-15 12:35 UTC
