# RF Chain Modeling
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18145791.svg)](https://doi.org/10.5281/zenodo.18145791)

**RF Signal Simulation and Analysis Framework**

`RF_chain_modeling` is a Python framework designed to model, simulate, and analyze Radio Frequency (RF) chains. It allows you to cascade various RF components—such as amplifiers, filters, cables, and antennas—to predict system performance metrics like Gain, Noise Figure (NF), and non-linearities (**OP1dB**, **IP3**, **IP2**).

## Key Features

*   **Component Modeling:** built-in models for Attenuators, Amplifiers, Cables, Filters (HighPass, BandPass, LowPass), and Antennas.
*   **Data-Driven Components:** Import component characteristics from CSV/TSV data files (e.g., S-parameters or measured gain/NF).
*   **Signal Processing:** Simulation of time-domain and frequency-domain signals with thermal noise injection.
*   **Performance Assessment:**
    *   **Gain & Phase:** Frequency response analysis.
    *   **Noise Figure (NF):** Cascaded noise analysis.
    *   **Non-Linearities:** Automatic assessment of 1dB Compression Point (P1dB) and Intercept Points (IIP3/OIP3, IIP2/OIP2).

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dunaar/RF_chain_modeling.git
    cd RF_chain_modeling
    ```

2.  **Install dependencies:**
    The project relies on standard scientific Python libraries.
    ```bash
    pip install -r requirements.txt
    ```
    *Required libraries: `numpy`, `scipy`, `matplotlib`, `tqdm`*

## 📖 Usage Examples

### 1. Modeling a Full RF Chain
The script `rf_chains/rf_chain_example.py` demonstrates how to create a complete transmission chain combining a signal generator, an antenna, filters, and amplifiers.

**How to run:**
```bash
python -m RF_chain_modeling.rf_chains.rf_chain_example
```

**What it does:**
*   **Signal Generation:** Creates a broad-band signal (40 GHz bandwidth) with thermal noise and three specific tones (at 3, 11, and 17 GHz).
*   **Chain Definition:**
    1.  **Antenna:** Defines frequency-dependent gain.
    2.  **High-Pass Filter:** Cutoff at 6 GHz.
    3.  **LNA (Low Noise Amplifier):** Gain 16 dB, NF 3 dB.
    4.  **Attenuator:** 5 dB attenuation.
    5.  **Power Amplifier:** Gain 20 dB, OIP3 40 dBm.
    6.  **RF Cable:** 10m length with frequency-dependent loss.
    7.  **Band-Pass Filter:** 9-12 GHz passband.
*   **Simulation:** Processes the signal through the chain and plots the Time Domain and Frequency Spectrum before and after processing.
*   **Assessment:** Automatically characterizes the Gain, Noise Figure, and Intercept points (IP2, IP3) of the entire chain and individual components.

---

### 2. Modeling a Specific Component from Data
The script `rf_components/zvq_183_s_plus.py` shows how to model a specific commercial component (Mini-Circuits ZVQ-183+ amplifier) using measured data stored in a TSV file.

**How to run:**
```bash
python -m RF_chain_modeling.rf_components.zvq_183_s_plus
```

**What it does:**
*   **Data Import:** Reads `zvq_183_s_plus.tsv` using the `CSVDataTable` utility to parse Frequency, Gain, Phase, NF, and OIP3 data.
*   **Component Creation:** Instantiates an `RF_Modelised_Component` using the real-world data points.
*   **Characterization:**
    *   Interpolates performance metrics across the frequency band (10 MHz - 20 GHz).
    *   Calculates and plots **Gain**, **Phase**, and **Noise Figure** vs. Frequency.
    *   Assesses linearity (P1dB, IP3) at specific test frequencies.

## 📂 Project Structure

*   `rf_utils/`: Core utilities for signal processing (`Signals` class), component base classes, and CSV handling.
*   `rf_components/`: Definitions for specific or generic RF components.
*   `rf_chains/`: Scripts defining and simulating complete RF chains.
*   `rf_subchains/`: Reusable sub-blocks of RF components.

## 📝 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

**Author:** Pessel Arnaud
**Version:** 0.1 (2026-01-03)

## Citing

If you use this software in your research, please cite it using the following DOI:

> Pessel, A. (2026). RF_chain_modeling (v0.1.0-beta.1). Zenodo. https://doi.org/10.5281/zenodo.18145792

