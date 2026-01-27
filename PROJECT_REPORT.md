# Maritime Boundary Alert System - Project Report

## 1. Project Overview
**Title:** Machine Learning-Based Smart Alert System for Preventing Maritime Boundary Crossing
**Target Audience:** Tamil Nadu Fishermen
**Goal:** To prevent international boundary crossings and subsequent arrests by providing intelligent, real-time alerts *before* a violation occurs.

## 2. Problem Statement
Fishermen often cross the International Maritime Boundary Line (IMBL) unknowingly due to checking GPS coordinates only periodically or lacking clear visual cues at sea. Existing solutions are often:
- **Reactive**: Alerting only *after* the boundary is crossed.
- **Passive**: Simple GPS logs without intelligent trajectory analysis.
- **Inaccessible**: Lacking local language (Tamil) support or intuitive audio cues.

## 3. Solution Architecture
This project uses a **Two-Layer Hybrid Approach** to move from reactive alerting to proactive prevention.

### Layer 1: Reactive Zone Classification (The "Where am I?" Layer)
- **Goal**: Identify the vessel's current safety status.
- **Method**: The sea is divided into three zones:
    1.  **Safe Zone**: Far from the boundary (> 5km).
    2.  **Caution Zone**: Near the boundary (2km - 5km).
    3.  **Danger Zone**: Immediate proximity (< 2km) or crossing.
- **Algorithm**: **K-Nearest Neighbors (KNN)** (or Logistic Regression). It takes current coordinates (Latitude/Longitude) and classifies them into one of the three zones with high accuracy.

### Layer 2: Proactive Trajectory Forecasting (The "Where am I going?" Layer)
- **Goal**: Predict where the vessel will be in the next 30-60 minutes.
- **Method**: Analyzes the vessel's movement history (last 5 GPS points) to detect speed and bearing.
- **Algorithm**: **Long Short-Term Memory (LSTM)** Network. This is a type of Recurrent Neural Network (RNN) excellent for time-series data.
- **Outcome**: Calculates a "Time to Intercept". If the vessel is in the Safe Zone but moving swiftly towards the border, the system triggers a **Pre-emptive Alert** ("You will cross in 10 minutes, turn back!").

## 4. Technical Implementation

### Tech Stack
- **Language**: Python 3.13
- **Web Framework**: Flask (Backend)
- **Doardboard UI**: HTML5 + Leaflet.js (Interactive Maps) + CSS3
- **Machine Learning**: 
    - `scikit-learn` (KNN Zone Classifier)
    - `tensorflow`/`keras` (LSTM Forecaster)
- **Data Handling**: `pandas`, `numpy`, `geopandas`
- **Audio**: `gTTS` (Google Text-to-Speech) for generating Tamil alerts.

### System Modules
1.  **Data Generator (`src/data_generator.py`)**: 
    - Since real-time sensitive military/fishing data is unavailable, we created a simulator.
    - Generates synthetic GPS tracks that mimic fishing boats drifting or moving towards the boundary.
2.  **AI Models (`src/models.py`)**:
    - `ZoneClassifier`: Predicts instant risk level.
    - `TrajectoryForecaster`: Predicts future position.
3.  **Alert System (`src/alert_system.py`)**:
    - Triggers Tamil voice messages: *"Echarikkai! Ellai thandum abayam."* (Warning! Danger of crossing boundary).
    - Plays distinct audio for Caution vs Danger.
4.  **Web Dashboard (`app.py` & `static/js/main.js`)**:
    - Displays live vessel position on a map.
    - Visualizes the "Danger Zone" as a red strip along the border.
    - Flashes visual warnings when alerts are triggered.

## 5. How It Works (Simulation)
1.  **Initialization**: The system loads pre-trained AI models.
2.  **Real-time Loop**: 
    - Every second, the system generates a new GPS point (simulating a moving boat).
    - **Check 1**: Is the point in the Danger Zone? -> **Alert**.
    - **Check 2**: Is the forecasted path leading to the border? -> **Pre-emptive Warning**.
3.  **User Feedback**:
    - **Audio**: Plays Tamil warnings through speakers.
    - **Visual**: The Dashboard Map flashes Red.

## 6. How to Run
1.  **Install Dependencies**: `pip install -r requirements.txt`
2.  **Train Models**: `python -m src.train`
3.  **Start Dashboard**: `python app.py`
4.  **View**: Open browser to `http://127.0.0.1:5000`

## 7. Future Enhancements
- **Hardware**: Deploy on a rugged Raspberry Pi device on actual boats.
- **Connectivity**: Use LoRaWAN or Satellite for shore communication (since mobile networks fail at sea).
- **Social**: Integration with Fisheries Department for emergency SOS.
