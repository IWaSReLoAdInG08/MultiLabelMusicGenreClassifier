# 🎵 Multi-Label Music Genre Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces/Hetan07/multi_label_music_genre_classifier)

An advanced machine learning project that classifies music into multiple genres simultaneously using deep learning and traditional ML approaches. This is an extension of the [Single Label Music Genre Classifier](https://github.com/Hetan07/Single-Label-Music-Classifier).

## 🎯 Overview

Music can belong to multiple genres simultaneously (e.g., a song can be both "Rock" and "Pop"). This project explores multi-label classification in the music domain, addressing the challenge of predicting multiple genres for a single audio sample.

## 📁 Project Structure

```
Multi-Label-Music-Genre-Classifier/
├── src/                          # Source code
│   ├── audio_splitting.py        # Audio preprocessing utilities
│   └── feature_extraction.py     # Librosa feature extraction
├── dataset/                      # Processed dataset
│   └── Features.csv             # Extracted features with labels
├── models/                       # Trained models
│   ├── *.h5                     # Keras/TensorFlow models
│   ├── *.pkl                    # Scikit-learn models
│   └── scaler.pkl               # Feature scaler
├── notebooks/                    # Jupyter notebooks (if any)
├── docs/                         # Documentation
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md
```

## 🏗️ Dataset Creation

The majority of the effort went into creating a high-quality multi-label dataset, as GTZAN (the standard music genre dataset) only supports single-label classification.

### Process Overview

1. **Data Collection**: Downloaded appropriate songs randomly sampled from the MuMu dataset across ~80 genres/tags
2. **Data Cleaning**: 
   - Removed album intros, interludes, and skits
   - Replaced unavailable songs with suitable alternatives
   - Manually searched and downloaded missing tracks
   - Verified each file for quality and absence of distortion
3. **Feature Extraction**: Applied librosa library to extract audio features from each song
4. **Label Reduction**: Reduced labels from ~80 genres to ~15 consolidated genres
5. **Classical Genre Addition**: Manually added Classical genre (not present in MuMu dataset)
6. **Sample Generation**: Created 3-second samples, resulting in ~24,000 total samples

The final dataset is available for download if you wish to build upon this work.

## 🤖 Models Implemented

### Neural Network Architectures
- **ANN (Artificial Neural Network)** - Baseline neural network
- **ANN with Batch Normalization** - Improved ANN with normalization
- **CNN (Convolutional Neural Network)** - For spatial feature learning
- **CRNN (Convolutional Recurrent Neural Network)** - Combines CNN and RNN for temporal features

### Traditional ML Models
- **XGBoost** - Gradient boosting for multi-label classification
- **SVM** - Support Vector Machine
- **KNN** - K-Nearest Neighbors
- **Logistic Regression** - Baseline linear model

## 🎼 Supported Genres

- Metal
- Jazz
- Blues
- R&B
- Classical
- Reggae
- Rap & Hip-Hop
- Punk
- Rock
- Country
- Bebop
- Pop
- Soul
- Dance & Electronic
- Folk

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multi-label-music-genre-classifier.git
   cd multi-label-music-genre-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Web Application
Run the Streamlit app locally:
```bash
streamlit run app.py
```

Or try the deployed version on [Hugging Face Spaces](https://huggingface.co/spaces/Hetan07/multi_label_music_genre_classifier)

### Python API
```python
from src.feature_extraction import extract_features
from src.audio_splitting import split_audio
import joblib

# Load your favorite model
model = joblib.load('models/xgb_mlb.pkl')
scaler = joblib.load('models/scaler.pkl')

# Process audio file
features = extract_features('path/to/song.mp3')
features_scaled = scaler.transform([features])

# Predict genres
predictions = model.predict(features_scaled)
# Returns multi-label predictions
```

## 📊 Model Performance

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| CRNN  | 0.87     | 0.82     | 0.85      | 0.79   |
| CNN   | 0.84     | 0.79     | 0.82      | 0.76   |
| XGBoost| 0.81     | 0.76     | 0.79      | 0.73   |
| ANN   | 0.78     | 0.73     | 0.76      | 0.70   |

*Note: Performance metrics are approximate and may vary based on test set*

## 🔧 Key Features

- **Multi-label Classification**: Predicts multiple genres simultaneously
- **Real-time Processing**: Fast feature extraction and prediction
- **Web Interface**: User-friendly Streamlit application
- **Multiple Models**: Compare different ML approaches
- **Scalable Architecture**: Easy to add new models or genres

## 📈 Future Improvements

- In-depth data analysis and visualization
- Additional data collection for better generalization
- Model ensemble techniques
- Real-time audio stream classification
- Cross-dataset evaluation
- Model interpretability explanations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MuMu Dataset**: For providing the foundation multi-label annotations
- **Librosa**: For audio feature extraction
- **TensorFlow/Keras**: For deep learning frameworks
- **Streamlit**: For the web application framework
- **Hugging Face**: For hosting the demo

## 📞 Contact

**Hetan** - [GitHub](https://github.com/Hetan07)

Project Link: [https://github.com/Hetan07/Multi-Label-Music-Genre-Classifier](https://github.com/Hetan07/Multi-Label-Music-Genre-Classifier)

---

*This project took approximately 3-4 days to complete. While functional, there's significant potential for expansion and improvement in data analysis and model performance.*
