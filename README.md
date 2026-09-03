# paddy-yield-estimation-dashboard
Spatio-temporal yield prediction and trend analytics dashboard for Malaysian paddy estate management.
Used for precision agriculture system for weed segmentation and spatiotemporal yield estimation, built for Malaysian paddy estates.

---

## Overview

This dashboard is an end-to-end system for Malaysian paddy estates, combining computer vision and machine learning to segment paddy from weeds in drone imagery and estimate yield for the MR297 paddy variety.

### Key Features

| Feature | Description |
|---------|-------------|
| Drone Image Analysis | Upload drone images to segment paddy from weeds |
| Yield Prediction | Estimate plot-level harvest tonnage from five input parameters |
| Interactive Dashboard | Five tabs covering the full analysis workflow |
| TRL Tracking | Tracks progress from TRL 4 to TRL 6 |
| Data Analytics | Trends, correlations, and data export for further analysis |

### Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Segmentation mIoU | 86.2% | > 85% (met) |
| Yield Prediction Error | 6.8% | < 10% (met) |
| R-squared (R²) | 0.9808 | — |
| Mean Absolute Error | 0.0657 MT | ± 66 kg |

---

## Project Objectives

| Objective | Description | Status |
|-----------|-------------|--------|
| Objective 1 | Build a MobileNetV2-UNet model for paddy vs. weed segmentation (mIoU > 85%) | Completed |
| Objective 2 | Build an XGBoost regressor for yield prediction (error < 10%) | Completed |
| Objective 3 | Validate the framework on a UAV platform (TRL 4 → 6) | In progress |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | Streamlit |
| Language | Python 3.14+ |
| Visualization | Plotly, Matplotlib |
| Data Processing | Pandas, NumPy |
| Segmentation | Classical CV (ExG + Otsu) / MobileNetV2-UNet |
| Prediction | Formula-based / XGBoost |
| Image Processing | Pillow, SciPy |
| Deployment | Local / Streamlit Cloud |

---

## Project Structure

