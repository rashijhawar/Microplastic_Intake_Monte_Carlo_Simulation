# Monte Carlo Simulation: Estimating Human Microplastic Intake through Food, Water, and Air
## Team Members: Rashi Jhawar and Tejal Bhansali

This project simulates daily human intake of microplastics (MP) through air, food, and water across different countries and dietary patterns using a Monte Carlo simulation. We have also tried to identify how each of the sources contributes to the total microplastic intake.

---

## 📊 Hypothesis 1

- Null Hypothesis: There is no significant difference between microplastic consumption between developed and developing countries.
- Alternative Hypothesis: There is a significant difference between microplastic consumption between developed and developing countries.

## 📊 Hypothesis 2

- Null Hypothesis: There is no significant difference in microplastic intake across different diet groups.
- Alternative Hypothesis: There is a significant difference in microplastic intake across different diet groups.

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
- Total microplastic exposure occurs only through 3 pathways: **food, water and air.**
---

## 📁 Project Structure
```
.
├── data_utils.py         # Contains functions for loading and merging data
├── simulation.py         # Contains functions for Monte Carlo simulation
├── visualizations.py     # Contains code for generating plots and visual summaries of simulation results
├── main.py               # Main execution script

---
## 🧪 Installation & Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/rashijhawar/Microplastic_Intake_Monte_Carlo_Simulation.git
   cd Microplastic_Intake_Monte_Carlo_Simulation

2. Set up a Python evnironment:
    ```bash
   python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install dependencies:
    ```bash
   pip install -r requirement.txt
   
---

## How to Run
Once your environment is set up and dependencies are installed, run the main script:
```python3 main.py