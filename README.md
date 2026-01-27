# 🚢 Maritime Boundary Alert System

**A Machine Learning-Based Smart Alert System for Preventing Maritime Boundary Crossings**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Framework](https://img.shields.io/badge/Flask-Web%20Dashboard-green)
![AI](https://img.shields.io/badge/Model-LSTM%20%2B%20KNN-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
This project addresses the critical issue of fishermen unknowingly crossing International Maritime Boundary Lines (IMBL). It uses a **Two-Layer Hybrid Architecture** to provide real-time, proactive warnings.

Instead of just alerting *after* a crossing (Reactive), this system forecasts the vessel's trajectory and warns *before* a violation occurs (Proactive).

## 🚀 Key Features
*   **Layer 1: Zone Classification**: Uses **KNN** to instantly classify the vessel's location into **Safe**, **Caution**, or **Danger** zones.
*   **Layer 2: Trajectory Forecasting**: Uses **LSTM (Deep Learning)** to predict the vessel's future path and estimate "Time to Intercept" the boundary.
*   **🗣️ Tamil Voice Alerts**: Integrated Text-to-Speech (TTS) provides clear audio warnings in the local language (*"Echarikkai!"*).
*   **🗺️ Real-time Dashboard**: A web-based interface (Leaflet.js) visualizing the vessel, boundary line, and predictive path.

## 🛠️ Tech Stack
*   **Language**: Python
*   **Web Framework**: Flask
*   **Frontend**: HTML5, CSS3, JavaScript, Leaflet.js
*   **Machine Learning**: TensorFlow (Keras), Scikit-Learn, NumPy, Pandas
*   **Audio**: gTTS (Google Text-to-Speech)

## ⚙️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Antony28090/Maritime-Alert-System.git
    cd Maritime-Alert-System
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Models** (First time only)
    ```bash
    python -m src.train
    ```

4.  **Run the Dashboard**
    ```bash
    python app.py
    ```

5.  **Access the Interface**
    Open your browser and navigate to: `http://127.0.0.1:5000`

## 📂 Project Structure
```
Maritime-Alert-System/
├── src/
│   ├── alert_system.py   # Handles Audio/TTS
│   ├── data_generator.py # Simulates GPS data
│   ├── models.py         # LSTM & KNN Implementations
│   └── train.py          # Model training script
├── static/               # CSS & JS for Dashboard
├── templates/            # HTML Template
├── app.py                # Flask Backend & Simulation Loop
└── requirements.txt      # Dependencies
```

## 🔮 Future Enhancements
*   Hardware integration with Raspberry Pi & GPS Modules.
*   LoRaWAN/Satellite communication for deep-sea alerts.
*   Mobile App integration for fishermen.

---
*Created by [Antony Franklin](https://github.com/Antony28090)*
