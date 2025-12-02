# Brazil Extreme Rainfall Intensity Forecaster (B-ERIF): Recalibrating IFD Curves

**Goal.** Use machine learning (RF, XGBoost, MLP, LSTM, VAE) to:
1. Learn the relationship between daily rainfall, time features, and extremes in Rio Grande do Sul using CHIRPS (2000–2024).
2. Recalibrate observed Intensity–Duration–Frequency (IDF) curves and compare them with the May 2024 flood event.

This notebook uses CHIRPS daily rainfall over Rio Grande do Sul (Brazil) to:
- build a historical database (2000–2024) and isolate the May 2024 flood window;
- train several machine learning models to predict extreme rainfall;
- train a generative VAE on annual rainfall maxima;
- recalibrate IDF curves using the best-performing models.

**Author:** Sarah Mohanna Araújo  
**Course:** EAEEE4000 – Machine Learning for Environmental Engineering – Final Project

