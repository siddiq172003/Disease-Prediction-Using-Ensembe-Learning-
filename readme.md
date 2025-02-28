# Disease Prediction Using Ensemble Learning
## Overview
This repository contains a machine learning-based disease prediction chatbot implemented using Python, Streamlit, and scikit-learn. The system predicts potential diseases based on user-reported symptoms and provides descriptions and precautions for the predicted conditions. It uses a Decision Tree Classifier trained on symptom-disease datasets and includes an interactive web interface powered by Streamlit.

The project leverages multiple datasets, including symptom descriptions, severity scores, precaution recommendations, and training data mapping symptoms to diseases. The goal is to assist users in identifying possible health conditions based on their symptoms and encourage them to seek professional medical advice when necessary.

## Features
- Symptom-Based Prediction: Users input symptoms, and the system predicts a likely disease using a trained Decision Tree Classifier.
- Interactive UI: Built with Streamlit for an easy-to-use web interface.
- Disease Information: Provides descriptions and precautionary measures for predicted diseases.
- Severity Assessment: Evaluates symptom severity and duration to suggest whether a doctor’s consultation is needed.
- Extensible: Supports additional datasets and machine learning models (e.g., SVM).

## Repository Structure
disease-prediction-chatbot/
│
├── data/
│   ├── Training.csv              # Training dataset mapping symptoms to diseases
│   ├── Testing.csv               # Testing dataset for model validation
│   ├── symptom_Description.csv   # Descriptions of diseases
│   ├── symptom_precaution.csv    # Precautions for each disease
│   ├── Symptom_severity.csv      # Severity scores for symptoms
│
├── src/
│   ├── svastya_chatbot.py        # Simple prediction chatbot implementation
│   ├── prana_chatbot.py          # Advanced chatbot with decision tree traversal
│
├── README.md                     # Project documentation (this file)
└── requirements.txt              # Python dependencies

## Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)
- A web browser (for running the Streamlit app)

## Installation
1. Install Dependencies:
```python
pip install -r requirements.txt
```
2. Prepare Data:
```
- Ensure all CSV files (Training.csv, Testing.csv, symptom_Description.csv, symptom_precaution.csv, Symptom_severity.csv) are placed in the data/ directory.
- Update file paths in the scripts (svastya_chatbot.py and prana_chatbot.py) if necessary to match your local setup.
```
## Usage
### Running the Simple Chatbot (Prana)
1. Navigate to the src/ directory: 
``` python
cd src
```
2. Run the Streamlit app:
```
streamlit run Prana_chatbot.py
```
3. Open your browser and go to http://localhost:8501.
4. Enter your name, select symptoms, specify the number of days, and click "Predict" to get a disease prediction.

## Datasets
- Training.csv & Testing.csv: Contain symptom-disease mappings (binary indicators for symptoms and corresponding prognosis).
- symptom_Description.csv: Descriptions of diseases for user education.
- symptom_precaution.csv: Recommended precautions for each disease.
- Symptom_severity.csv: Severity scores (1-7) for symptoms to assess condition urgency.

## Model Details
- Algorithm: Decision Tree Classifier (with an optional SVM implementation).
- Training: The model is trained on Training.csv with a 67-33 train-test split.
- Evaluation: Accuracy is assessed using cross-validation and test set predictions.

## Contributing
1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes and commit (git commit -m "Add feature").
4. Push to your branch (git push origin feature-branch).
5. Open a pull request.

## Limitations
- The system is not a substitute for professional medical advice.
- Predictions are based on limited symptom data and may not account for rare conditions or comorbidities.
- Accuracy depends on the quality and completeness of the training data.

## License
- This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Built with -Streamlit-, -scikit-learn-, and -Pandas-.
- Inspired by real-world health informatics applications.

## Notes for Customization
1. Replace your-username in the Git clone URL with your actual GitHub username.
2. Add a requirements.txt file to your repository with the following content:
```
streamlit
pandas
scikit-learn
numpy

```
- Update file paths in the README if your directory structure differs.
- If you have a specific project name (e.g., "Svastya" or "Prana"), replace "Disease Prediction Chatbot" with it.
