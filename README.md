# RF Chain Modeling
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18145791-blue)](https://doi.org/10.5281/zenodo.18145791)

![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=matplotlib&logoColor=black)
![License](https://img.shields.io/github/license/dunaar/RF_chain_modeling)

**RF Signal Simulation and Analysis Framework**

`rf_chain_modeling` is a Python framework designed to model, simulate, and analyze Radio Frequency (RF) chains. It allows you to cascade various RF components—such as amplifiers, filters, cables, and antennas—to predict system performance metrics like Gain, Noise Figure (NF), and non-linearities (**OP1dB**, **IP3**, **IP2**).

## Key Features

*   **Component Modeling:** built-in models for Attenuators, Amplifiers, Cables, Filters (HighPass, BandPass, LowPass), and Antennas.
*   **Data-Driven Components:** Import component characteristics from CSV/TSV data files (e.g., S-parameters or measured gain/NF).
*   **Signal Processing:** Simulation of time-domain and frequency-domain signals with thermal noise injection.
*   **Performance Assessment:**
    *   **Gain & Phase:** Frequency response analysis.
    *   **Noise Figure (NF):** Cascaded noise analysis.
    *   **Non-Linearities:** Automatic assessment of 1dB Compression Point (P1dB) and Intercept Points (IIP3/OIP3, IIP2/OIP2).

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/dunaar/RF_chain_modeling.git
cd RF_chain_modeling
```

### 2. Install the package
The project uses `pyproject.toml` for dependency management. You can install it using modern environment managers like `uv` (recommended) or standard `pip`.

**Using uv (Recommended):**
```bash
# 1. Create a virtual environment
uv venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# 2. Sync the project and install the package with dev dependencies
uv sync --extra dev
```

**Using standard pip:**
```bash
# Install the package in editable mode
pip install -e .

# If you want to contribute or run tests, install the dev dependencies:
pip install -e ".[dev]"
```

Required libraries automatically installed: `numpy`, `scipy`, `matplotlib`, `tqdm`.

## 📖 Usage Examples

### 1. Running the Main Entry Point
The project serves a primary chain example which can be executed directly via the package's main entry point. 

**Standard execution (module mode via `__main__.py`):**
```bash
python3 -m rf_chain_modeling
```
**Using uv:**
```bash
uv run python -m rf_chain_modeling
```

*(Alternatively, you can run the `main.py` script located at the repository root: `uv run python main.py`)*

---

### 2. Modeling a Full RF Chain
The script `rf_chain_example.py` demonstrates how to create a complete transmission chain combining a signal generator, an antenna, filters, and amplifiers.

**Standard execution (module mode):**
```bash
python3 -m rf_chain_modeling.examples.rf_chain_example
```
**Using uv:**
```bash
uv run python -m rf_chain_modeling.examples.rf_chain_example
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

### 3. Modeling the RF Core Classes
The script `rf_modeling_example.py` demonstrates the core signal modeling features independently from a full RF chain.

**Standard execution (module mode):**
```bash
python3 -m rf_chain_modeling.examples.rf_modeling_example
```
**Using uv:**
```bash
uv run python -m rf_chain_modeling.examples.rf_modeling_example
```

**What it does:**
*   Creates a signal over a wide RF bandwidth.
*   Adds thermal noise and several tones with different amplitudes and phases.
*   Computes RMS-related quantities in time and frequency domains.
*   Plots temporal and spectral representations of the signal.

---

### 4. Modeling a Reusable RF Subchain
The script `rf_subchain_example.py` provides an additional example entry point for reusable RF subchain workflows.

**Standard execution (module mode):**
```bash
python3 -m rf_chain_modeling.examples.rf_subchain_example
```
**Using uv:**
```bash
uv run python -m rf_chain_modeling.examples.rf_subchain_example
```

---

### 5. Modeling a Specific Component from Data
The script `rf_components/zvq_183_s_plus.py` shows how to model a specific commercial component (Mini-Circuits ZVQ-183-S+ amplifier) using measured data packaged within the module.

**Standard execution (module mode):**
```bash
python3 -m rf_chain_modeling.rf_components.zvq_183_s_plus
```
**Using uv:**
```bash
uv run python -m rf_chain_modeling.rf_components.zvq_183_s_plus
```

**What it does:**
*   **Data Import:** Reads `zvq_183_s_plus.tsv` securely via `importlib.resources`.
*   **Component Creation:** Instantiates an `RF_Modelised_Component` using the real-world data points.
*   **Characterization:**
    *   Interpolates performance metrics across the frequency band (10 MHz - 20 GHz).
    *   Calculates and plots **Gain**, **Phase**, and **Noise Figure** vs. Frequency.
    *   Assesses linearity (P1dB, IP3) at specific test frequencies.

### 6. Running the Tests
The project includes a comprehensive test suite (unit tests and end-to-end integration tests) located in the `tests/` directory.

If you installed the package with the `[dev]` dependencies, you can run the test suite using `pytest`:

**Standard execution:**
```bash
pytest
# Or to see verbose output:
pytest -vl
```
**Using uv:**
```bash
uv run pytest
# Or to see verbose output:
uv run pytest -vl
```

## 📂 Project Structure

The package source code is located under the `src/rf_chain_modeling/` directory:

*   `examples/`: Demonstration scripts showing how to use the framework (e.g., complete RF chain, specific components).
*   `rf_utils/`: Core utilities for signal processing (`Signals` class), abstract base component classes, and CSV data handling.
*   `rf_components/`: Definitions for specific or generic RF components (e.g., amplifiers, cables, filters), including packaged `.tsv` data.
*   `rf_chains/`: Definitions and simulations of complete cascaded RF chains.
*   `rf_subchains/`: Reusable sub-blocks of RF components.

The test suite is located under `tests/` and currently includes:
*   `test_10_rf_modeling.py`
*   `test_20_components.py`
*   `rf_components/test_21_zvq_183_s_plus.py`

## 📝 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

**Author:** Pessel Arnaud

## 📚 Citing

If you use this software in your research, please cite it using the following DOI (as defined in the `CITATION.cff` release):

> Pessel, A. (2026). rf_chain_modeling (v0.1.2.dev1). Zenodo. https://doi.org/10.5281/zenodo.18145792

## 🤝 Contributing

When contributing to this project, you must strictly adhere to the Functional Requirements (`FR-STD-01` to `FR-STD-07`) defined in `pyproject.toml`:

*   **Language (`FR-STD-01`)**: All source code, docstrings, variable names, and inline comments must be written entirely in **English**.
*   **Documentation (`FR-STD-02`, `FR-STD-04`)**: Use the **Google docstring style** for all public classes and methods.
*   **Logging (`FR-STD-05`)**: Do not use `print()` statements in production code; use the `logging` module.
*   **Naming (`FR-STD-06`)**: Strictly preserve and respect underscore-based naming conventions in all Python identifiers (e.g., `rf_chain_modeling`, `im2___power`).
*   **Formatting (`FR-STD-07`)**: Maintain existing vertical alignments for assignment operators and dictionary definitions.