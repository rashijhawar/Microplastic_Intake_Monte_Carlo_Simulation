# Microplastic Intake Simulation

This project simulates daily human intake of microplastics (MP) through food consumption across different countries and dietary patterns using Monte Carlo simulations. It uses statistical modeling, including log-normal and PERT distributions, to estimate variability in intake and exposure, and visualizes results using violin and stacked bar plots.

---

## 📊 Hypothesis 1

- Null Hypothesis: There is no significant difference between microplastic consumption between developed and developing countries.
- Alternative Hypothesis: There is significant difference between microplastic consumption between developed and developing countries.

## 📊 Hypothesis 2

- Null Hypothesis: There is no significant difference in microplastic intake across different diet groups.
- Alternative Hypothesis: There is significant difference in microplastic intake across different diet groups.

---

## 📊 Project Features

- Monte Carlo simulations of MP intake using log-normal distributions
- Per-country and per-diet analysis of intake
- Violin plots to visualize MP intake distributions
- Stacked bar charts to show food item contributions
- Doctests for simulation and plotting functions
- Supports custom Excel datasets for intake and concentration

---

## 🧮 Assumptions

- **Intake of food, water, and air** follows a **log-normal distribution**
- **Weight of a microplastic particle** follows a **modified PERT distribution**
- **Standard deviation** for log-normal distribution is assumed to be **25% of the mean**
- Total microplastic exposure occurs only through 3 pathways - **food, water and air.**
---

## 📁 Project Structure

---
## 🧪 Installation & Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/mp-intake-simulation.git
   cd mp-intake-simulation

2. Set up a Python evnironment:
    ```bash
   python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install dependencies:
    ```bash
   pip install -r requirement.txt
   
---

## How to Run


