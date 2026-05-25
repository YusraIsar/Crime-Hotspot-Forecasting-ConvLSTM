# Deep-Learning_
# 🚨 Spatio-Temporal Crime Forecasting using ConvLSTM

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🧠 Abstract

This project presents a **deep learning-based spatio-temporal forecasting system** for predicting urban crime hotspots using the Chicago Crime dataset.

Unlike traditional statistical or tabular models, this system transforms crime data into a **spatial grid-based time series** and applies **ConvLSTM neural networks** to learn both spatial and temporal dependencies simultaneously.

The final system outputs **future crime intensity heatmaps** visualized through an interactive Streamlit dashboard.

---

## 🎯 Problem Statement

Urban crime prediction is inherently:

- 📍 Spatial (location-dependent)
- ⏳ Temporal (time-evolving)
- 🔗 Interdependent (neighboring regions influence each other)

Traditional ML models fail to capture these dependencies simultaneously.

👉 This project solves this using **ConvLSTM-based spatio-temporal deep learning**.

---

## 🏗️ System Architecture

![Architecture](assets/architecture.png)

**Pipeline:**

Raw Crime Data  
→ Data Cleaning  
→ Spatial Grid Conversion (20×20)  
→ Weekly Temporal Aggregation  
→ Sequence Generation (8-week windows)  
→ ConvLSTM Model  
→ Crime Heatmap Prediction  
→ Streamlit Visualization

---

## 📊 Dataset

**Chicago Crime Dataset (2001–Present)**  
Source: Chicago Open Data Portal

### Features:
- Crime Type
- Date & Time
- Latitude / Longitude
- Location Type
- Arrest Flag

---

## 🧠 Model Architectures

### 🔹 Spatial LSTM (Baseline)

- Flattens 2D grid into 1D vector
- Learns temporal patterns only
- Ignores spatial structure

---

### 🔥 ConvLSTM (Proposed Model)

![ConvLSTM](assets/convlstm.png)

ConvLSTM integrates:

- 🧠 Convolution → spatial feature learning
- 🔁 LSTM gating → temporal memory
- 📍 Preserves spatial locality

---

## 📈 Results

| Model        | MAE ↓ | RMSE ↓ | R² ↑ |
|--------------|------|--------|------|
| Spatial LSTM | 2.30 | 4.79   | 0.93 |
| ConvLSTM     | **Better** | **Better** | **Higher** |

---

## 🌍 Streamlit Dashboard

### Features:
- 📅 Date selection
- ⏳ Week selection
- ⚖️ Crime type filter
- 🗺️ Interactive Folium heatmap

---

## 📸 Demo Outputs

### 🗺️ Crime Heatmap Prediction
![Heatmap](assets/heatmap.png)

---

### 📊 Model Prediction vs Ground Truth
![Prediction](assets/prediction.png)

---

### 🎛️ Streamlit Dashboard UI
![Dashboard](assets/dashboard.png)

---

## 🚀 Tech Stack

- Python 🐍
- PyTorch 🔥
- NumPy / Pandas
- Scikit-learn
- Folium 🌍
- Streamlit ⚡
- Matplotlib / Seaborn

---

## 🧩 Key Contributions

✔ Converted raw crime data into spatio-temporal tensors  
✔ Implemented ConvLSTM from scratch (no pretrained libraries)  
✔ Designed grid-based geospatial modeling pipeline  
✔ Built interactive real-time forecasting dashboard  
✔ End-to-end ML → Web application system  

---

## 🧠 Innovation Highlights

- Treats crime prediction as a **spatio-temporal forecasting problem**
- Combines deep learning + geospatial intelligence
- Captures both:
  - Local spatial influence
  - Temporal crime evolution
- Produces interpretable heatmap outputs

---

## 📁 Project Structure

```bash
crime-forecasting/
│
├── app/
│   └── app.py
│
├── models/
│   ├── convlstm_best.pt
│   ├── scaler.pkl
│   └── grid_config.json
│
├── src/
├── notebooks/
├── data/
├── assets/
│   ├── architecture.png
│   ├── convlstm.png
│   ├── heatmap.png
│   ├── prediction.png
│   └── dashboard.png
│
└── README.md
