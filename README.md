# SeasFire: Subseasonal to Seasonal Global Wildfire Danger Forecasting

Wildfires are a growing global threat, impacting ecosystems, economies, and human livelihoods. 
While short-term wildfire predictions often leverage local weather conditions, forecasting fire 
activity weeks to months in advance remains a challenge due to the complex interplay of climate, 
vegetation dynamics, land surface processes, and human activities.

This challenge focuses on **subseasonal to seasonal wildfire forecasting** using the **SeasFire datacube** — a comprehensive spatiotemporal dataset designed for long-lead wildfire modeling.

You are invited to explore the dataset and develop data-driven models to improve our understanding and prediction of wildfire dynamics at extended lead times.

## Recommended Reading Material
- [SeasFire Datacube: A Dataset for Seasonal Wildfire Prediction](https://www.nature.com/articles/s41597-025-04546-3)
- [TeleViT: Teleconnection-Driven Transformers for Subseasonal-to-Seasonal Wildfire Forecasting (ICCV 2023 Workshop)](https://openaccess.thecvf.com/content/ICCV2023W/AIHADR/papers/Prapas_TeleViT_Teleconnection-Driven_Transformers_Improve_Subseasonal_to_Seasonal_Wildfire_Forecasting_ICCVW_2023_paper.pdf)
- [TeleViT: Teleconnection-Aware Vision Transformer for Subseasonal-to-Seasonal Wildfire Forecasting](https://arxiv.org/abs/2512.00089)
- [FireCastNet: Deep Learning for Wildfire Forecasting](https://www.nature.com/articles/s41598-025-30645-7)
- [U-Net Implementation for Spatiotemporal Prediction](https://arxiv.org/pdf/2211.00534)
---

## Jump in the notebook

All experiments are designed to run using **Google Colab**.

Click the badge below to launch the baseline notebook directly in Google Colab: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WinterSchool2026/ch01-seasfire/blob/main/seasfire_challenge_notebook.ipynb)

---

## Challenge Objectives

You are encouraged to investigate one (or more) of the following research directions:

### 1. Problem Formulation
It would make sense to move from a classification task to a more informative task.

Possible research questions:
- What would be the different approaches for building a model for that task?
- What are some metrics that are important for this formulation?

### 2. Sources of Regional Predictability

Wildfire predictability varies substantially across ecosystems and climate regimes. This track investigates where and why forecasting works better.

Possible research questions:

How does predictor importance vary across:
- Biomes (e.g., boreal forest, Mediterranean ecosystems, savannas)
- Climate zones (arid, temperate, tropical)
- Which regions exhibit stable predictability at longer lead times?
- Can we identify predictability regimes (fuel-limited vs climate-driven systems)?

### 3. Advanced Machine Learning & Deep Learning Approaches

This track explores new modeling strategies for spatiotemporal wildfire prediction.

Possible research questions:

- How can temporal-only predictors (e.g., OCI time series) be integrated into spatial models such as U-Net?
