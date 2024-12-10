# ❤️ Heart Disease Risk Prediction App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 🚀 An interactive web application for predicting 10-year coronary heart disease (CHD) risk using machine learning.

![App Demo](https://raw.githubusercontent.com/yourusername/heart-disease-predictor/main/demo.gif)

## ✨ Features

- 🔍 **Personalized Risk Assessment**: Get instant predictions for your 10-year CHD risk
- 📊 **Interactive Data Visualization**: Explore risk factors through dynamic charts
- 🤖 **AI-Powered Analysis**: Utilizing Random Forest algorithm with 85% accuracy
- 📱 **User-Friendly Interface**: Clean, intuitive design for easy navigation
- 📈 **Real-time Insights**: Immediate feedback and recommendations

## 🎯 Quick Start

### 🌐 Online Demo
Visit our live demo: [Heart Disease Predictor App](https://share.streamlit.io)

### 💻 Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

## 📊 Project Structure

```
heart-disease-predictor/
├── app.py                     # Main Streamlit application
├── model_development.py       # Model training script
├── exploratory_analysis.py    # Data analysis script
├── requirements.txt           # Project dependencies
├── framingham.csv            # Dataset
├── heart_disease_model_pipeline.joblib  # Trained model
└── README.md                 # Project documentation
```

## 🔬 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 85% |
| ROC AUC | 0.75 |
| Precision | 0.82 |
| Recall | 0.71 |

## 📱 App Screenshots

<table>
  <tr>
    <td><img src="screenshots/home.png" alt="Home Page" width="200"/></td>
    <td><img src="screenshots/prediction.png" alt="Prediction Page" width="200"/></td>
    <td><img src="screenshots/analysis.png" alt="Analysis Page" width="200"/></td>
  </tr>
</table>

## 🛠️ Technologies Used

- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) Python 3.8+
- ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) Streamlit
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) scikit-learn
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) Pandas
- ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) Plotly

## 📖 Dataset

The Framingham Heart Study dataset includes:
- 🏥 4,240 patient records
- ⏳ 10-year follow-up period
- 📈 15+ health parameters
- 🎯 Binary classification task

## 🚀 Features Used in Prediction

1. Age
2. Blood Pressure
3. Cholesterol Levels
4. Smoking Status
5. BMI
6. Heart Rate
7. Glucose Levels
8. And more...

## 💡 How to Use

1. Navigate to the "Risk Prediction" page
2. Enter your health information
3. Click "Predict Risk"
4. Get instant results and recommendations
5. Explore data insights and model performance

## ⚠️ Medical Disclaimer

This tool is for educational purposes only and should not replace professional medical advice. Always consult with healthcare providers for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - [GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- Framingham Heart Study for the dataset
- Streamlit team for the amazing framework
- All contributors and users of this application

## 📬 Contact

For questions and feedback:
- 📧 Email: your.email@example.com
- 🐦 Twitter: [@yourusername](https://twitter.com/yourusername)
- 💼 LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)

---

<p align="center">
  Made with ❤️ for better heart health
  <br>
  © 2024 Heart Disease Risk Predictor
</p> 