# Monte Carlo Simulation: Estimating Human Microplastic Intake through Food, Water, and Air
## Team Members: Rashi Jhawar and Tejal Bhansali

## 💡 Project Overview
This project simulates daily human intake of microplastics (MP) through air, food, and water across different countries and dietary patterns using a Monte Carlo simulation. We have also tried to identify how each of the sources contributes to the total microplastic intake.

---

## 🧩 Phase 1 of Monte Carlo Simulation: Design
### Random Variables
1. Daily per capita intake for 18 food categories ​(in g)
2. Microplastic concentration in a gram of different categories of food (in mg of microplastic/g of food)
3. Microplastics consumed through water (in mg)*
4. Microplastics consumed through air (in particles)*
5. Microplastic weight (in mg)

\* Microplastic consumption values represent daily consumption by a single individual

### Assumptions

- **Intake of food, water, and air** follows a **log-normal distribution**
- **Weight of a microplastic particle** follows a **modified PERT distribution**
- **Standard deviation** for log-normal distribution is assumed to be **25% of the mean**
- Total microplastic exposure occurs only through 3 pathways: **food, water and air.**

---

## ✅ Phase 2 of Monte Carlo Simulation: Validation
We ran the simulation for a subset of countries and found our results to be comparable with the published microplastic intake values for these countries.

<img width="600" alt="Image" src="https://github.com/user-attachments/assets/353a9480-a673-4862-9cde-4395d11a49f5" />

The convergence plot was as follows:

<img width="800" alt="Image" src="https://github.com/user-attachments/assets/c9d6d01d-2084-477d-b6a6-cab1123aff3a" />

---

## 🧪 Phase 3 of Monte Carlo Simulation: Experimentation

### 📊 Hypothesis 1

- Null Hypothesis: There is no significant difference between microplastic consumption between developed and developing countries.
- Alternative Hypothesis: There is a significant difference between microplastic consumption between developed and developing countries.

**Results:**

We observed that there was not much of a difference in microplastic consumption between developed and developing countries. Therefore, we concluded that microplastic intake did not depend on whether a country was developing or developed, and hence, we accept the null hypothesis.​

![Image](https://github.com/user-attachments/assets/fc829593-c6fc-4f10-9737-612a4fc38dc5)

**Some more visualizations from the simulation:**

This chart shows the top 15 countries with the highest daily microplastic intake per person, measured in milligrams per day.​

![Image](https://github.com/user-attachments/assets/d9640e63-3a7a-49a8-8b3b-46f59feb2814)

This chart helps give a view of the percentage contribution of each source to the total intake.

![Image](https://github.com/user-attachments/assets/56a7cedc-f1f2-4363-9dd0-c3f0af50915a)

Total per capita microplastic intake (through food, water, and air) in milligrams per day for each country plotted on the world map.

<img width="800" alt="Image" src="https://github.com/user-attachments/assets/684d4caf-ab18-4ed4-8a6d-6fa44ac23158" />


### 📊 Hypothesis 2

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

## ⚙️ Installation & Setup

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

## ▶️ How to Run
Once your environment is set up and dependencies are installed, run the main script:
```bash
python3 main.py