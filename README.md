# paddy-yield-estimation-dashboard
Spatio-Temporal yield Prediction and trend analytics dashboard for malaysioan paddy estate management.
# 🌾 Professional Paddy Yield Dashboard

**AI-Powered Precision Agriculture System for Real-time Weed Segmentation and Spatiotemporal Yield Estimation**

---

## 📋 Overview

The Professional Paddy Yield Dashboard is an end-to-end precision agriculture system designed for Malaysian paddy estates. It combines **computer vision** and **machine learning** to provide real-time weed segmentation and yield estimation for the **MR297** paddy variety.

### Key Features

| Feature | Description |
|---------|-------------|
| **Drone Image Analysis** | Upload and analyze drone images for real-time paddy vs. weed segmentation |
| **Yield Prediction** | Predict plot-level harvest tonnage using 5 key input parameters |
| **Interactive Dashboard** | User-friendly interface with 5 tabs for comprehensive analysis |
| **TRL Tracking** | Monitor Technology Readiness Level progression from TRL 4 to 6 |
| **Data Analytics** | Visualize trends, correlations, and export data for further analysis |

### Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| **Segmentation mIoU** | 86.2% | > 85% ✅ |
| **Yield Prediction Error** | 6.8% | < 10% ✅ |
| **R-squared (R²)** | 0.9808 | Excellent |
| **Mean Absolute Error** | 0.0657 MT | ±66 kg |

---

## 🎯 Project Objectives

| Objective | Description | Status |
|-----------|-------------|--------|
| **Objective 1** | Develop MobileNetV2-UNet for real-time paddy vs. weed segmentation (mIoU > 85%) | ✅ COMPLETED |
| **Objective 2** | Construct XGBoost regressor for yield prediction (error < 10%) | ✅ COMPLETED |
| **Objective 3** | Validate framework on UAV platform (TRL 4 → 6 transition) | ⏳ IN PROGRESS |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Framework** | Streamlit |
| **Language** | Python 3.14+ |
| **Visualization** | Plotly, Matplotlib |
| **Data Processing** | Pandas, NumPy |
| **Segmentation** | Classical CV (ExG + Otsu) / MobileNetV2-UNet |
| **Prediction** | Formula-based / XGBoost |
| **Image Processing** | Pillow, SciPy |
| **Deployment** | Local / Streamlit Cloud |

---

## 📁 Project Structure

