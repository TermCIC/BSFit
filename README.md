BSFit: Black Soldier Fly Growth Model Fitting Program

BSFit is a Python-based program designed to fit daily body width measurement data of black soldier fly (BSF) larvae to a logistic growth model. The program automates the process of data analysis by performing non-linear regression to estimate key growth parameters and generating visual outputs for model fitting and growth rate analysis.

Key Features of BSFit:

Logistic Model Fitting:

BSFit fits the provided daily body width measurements to the logistic growth model. This model captures the growth dynamics of BSF larvae by estimating key parameters such as asymptotic size (L), growth rate (K), and the inflection point (m), which represents the time of maximal growth rate.
Automatic Parameter Estimation:
Initial parameter values are calculated through linear regression using a logit transformation of the body width data. These initial estimates are then optimized through non-linear regression to ensure the best fit of the model to the observed data.
The program outputs the estimated model parameters, including L, K, m, and initial body size (W0), along with their standard errors and 95% confidence intervals.
Visual Output:

Model Fitting Plot:

BSFit generates a plot comparing the observed body width data to the predicted values from the logistic growth model. The observed data are shown as points, while the model predictions are displayed as a continuous curve over time. This plot helps users visually assess how well the model fits the experimental data.
AGR vs. RGR Plot: BSFit also calculates and plots the Absolute Growth Rate (AGR) and Relative Growth Rate (RGR) over time. AGR represents the change in body width per unit time, while RGR normalizes this change by the body size, offering insights into growth efficiency. Both metrics are plotted on the same graph, using different y-axes for clear comparison.
Data Export:
The program saves the estimated model parameters, their standard errors, and confidence intervals into a CSV file for easy review and further analysis.
The generated plots, including the model fitting and AGR vs. RGR graph, are exported as high-quality TIFF images, suitable for publication or presentation.
