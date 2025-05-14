# Monte Carlo Simulation: Estimating Human Microplastic Intake through Food, Water, and Air
## Team Members: Rashi Jhawar and Tejal Bhansali

## Project Overview
This project simulates daily human intake of microplastics (MP) through air, food, and water across different countries and dietary patterns using a Monte Carlo simulation. We have also tried to identify how each of the sources contributes to the total microplastic intake.

---

## Phase 1 of Monte Carlo Simulation: Design
### Random Variables
1. Daily per capita intake for 18 food categories ​(in g)
2. Microplastic concentration in a gram of different categories of food (in mg of microplastic/g of food)
3. Microplastics consumed through water (in mg)*
4. Microplastics consumed through air (in particles)*
5. Microplastic weight (in mg)

* Microplastic consumption values represent daily consumption by a single individual

### 🧮 Assumptions

- **Intake of food, water, and air** follows a **log-normal distribution**
- **Weight of a microplastic particle** follows a **modified PERT distribution**
- **Standard deviation** for log-normal distribution is assumed to be **25% of the mean**
- Total microplastic exposure occurs only through 3 pathways: **food, water and air.**

---

## Phase 2 of Monte Carlo Simulation: Validation
We ran the simulation for a subset of countries and found our results to be comparable with the published microplastic intake values for these countries.

![Image](https://github.com/user-attachments/assets/f240be05-7e62-47bd-9377-d3e46c9bdfcf)

The convergence plot was as follows:
<img width="468" alt="Image" src="https://github.com/user-attachments/assets/c9d6d01d-2084-477d-b6a6-cab1123aff3a" />

---

## Phase 3 of Monte Carlo Simulation: Experimentation

## 📊 Hypothesis 1

- Null Hypothesis: There is no significant difference between microplastic consumption between developed and developing countries.
- Alternative Hypothesis: There is a significant difference between microplastic consumption between developed and developing countries.

## 📊 Hypothesis 2

- Null Hypothesis: There is no significant difference in microplastic intake across different diet groups.
- Alternative Hypothesis: There is a significant difference in microplastic intake across different diet groups.

---

## 📁 Project Structure
```
.
├── data_utils.py         # Contains functions for loading and merging data
├── simulation.py         # Contains functions for Monte Carlo simulation
├── visualizations.py     # Contains code for generating plots and visual summaries of simulation results
├── main.py               # Main execution script
```
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
```python3 main.py```