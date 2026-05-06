# Provenance Aware Preprocessing for Time Series Data  - Algorithm

## Domain-Aware Time Series Imputation

A flexible Python framework for reconstructing missing values (NaN gaps) in time series data using adaptive, structure-aware interpolation methods.

The algorithm automatically selects different reconstruction strategies depending on:

- gap size
- signal behavior
- expected extrema density
- periodicity
- application domain

Supported domains include:

- eye tracking
- medical signals
- traffic data
- weather data
- water monitoring
- generic time series
  
## Features

- Automatic NaN gap detection
- Optional merging of nearby gaps
- Adaptive method selection
- Extrema-aware reconstruction
- Seasonal and periodic signal support
- Soft boundary protection against unrealistic values
- Detailed reconstruction logging

## Reconstruction Methods

Depending on the signal and gap size, the algorithm can use:

- Linear interpolation
- Polynomial fitting (AIC/BIC optimized)
- Cubic splines
- PCHIP interpolation
- Sinusoidal reconstruction
- Template-based reconstruction
- Seasonal reconstruction

---

## Project Status

**This algorithm is currently under active development and research.**

Features, reconstruction strategies, and parameter behavior may still change as the framework is continuously improved and evaluated on different time series domains.

  
