# DL & GenAI Course Project
Project Title : Smart MCQ Solver Challenge

Student Information :

    Name: Prashant Singh    
    Roll Number: 24f2007883
## Project Directory Structure

```text
├── data/                  # Contains the competition datasets
│   ├── train.csv          # Training dataset with answer labels
│   └── test.csv           # Test dataset used for inference and submission
│
├── notebooks/             # Jupyter notebooks used during different project milestones
│   ├── M1_NLP_Baseline.ipynb
│   ├── M2_Transformers_Embeddings.ipynb
│   ├── M3_RAG_Pipeline.ipynb
│   └── M4_M5_FineTuning_Ensemble.ipynb
│
├── scripts/               # Reusable and modular Python source code
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning, formatting, and tokenization
│   ├── engine.py               # Model training and validation pipeline
│   ├── inference.py            # Prediction and Kaggle submission generation
│   └── utils.py                # Utility functions and MAP@3 evaluation metric
│
├── reports/               # Contains project reports and presentations 
│
│
├── .gitignore             # Excludes unnecessary files from version control
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies required to reproduce results
```

Milestone-1 : Initial NLP Baseline Model 