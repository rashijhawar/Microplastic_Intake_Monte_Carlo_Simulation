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

**Results:**

The violin plots for top 20 countries clearly highlight the distribution and variation within each diet group. 
Notably, there’s a visible difference between animal-based and plant-based diet groups, suggesting a statistically significant impact of dietary choices on microplastic exposure.

![Image](https://github.com/user-attachments/assets/0311a938-c1c8-4dc6-8f69-d7afb9018092)


The below graph shows that seafood is the primary contributor to microplastic intake in the top 10 countries shown. 
Marine organisms ingest plastic waste dumped into the oceans, and microplastics accumulate in certain body parts. When these parts are consumed by humans, microplastics enter our bodies as well.
![Image](https://github.com/user-attachments/assets/a9f8d898-4bea-4bcd-8531-6c9f77c5d675)


Here’s where it gets interesting. In some countries, the microplastic intake among vegetarians and vegans is nearly comparable to that of omnivores and pescatarians.
This suggests that even though plant-based foods generally contain lower concentrations of microplastics, the sheer volume consumed — especially staples like grains, vegetables, and fruits — can significantly contribute to overall exposure.
![Image](https://github.com/user-attachments/assets/2b1f13f5-becf-4b46-8c21-67a2cf014e82)

Refined grains, shown in green, contribute to a large portion of the overall intake across many countries.
And that orange segment? That’s milk! A surprising but consistent contributor.
![Image](https://github.com/user-attachments/assets/820865d8-5fc5-4942-9b8d-7b9435aaadaa)

What this analysis reveals is important: diet group — vegetarian, vegan, or omnivore — determines your overall microplastic intake.



---
## 🚧 Limitations and Future Scope

* First, microplastics vary greatly in size, composition, and density, making it difficult to standardize their measurement and compare across studies.​
​
* Second, there is no single comprehensive study that evaluates all food sources uniformly, so our estimates are derived from a variety of independent reports with differing methodologies.​
​
* Third, this project does not account for nanoplastics, which are even smaller than microplastics and may pose greater health risks but are still largely unquantified in current literature.​
​
* Fourth, we have not considered microplastics in indoor and outdoor air, and bottled and tap water separately.​
​
* Lastly, our study was focused on 4 diet groups comprising of 18 food items in total, which can be broadened as a future scope.​

---
## 📚 References

* https://measurlabs.com/blog/microplastic-testing-standard-iso-24187/ ​
* https://ehp.niehs.nih.gov/doi/full/10.1289/EHP8936 ​
* https://pubs.acs.org/doi/epdf/10.1021/acs.est.4c00010?ref=article_openPDF ​
* https://www.fda.gov/food/nutrition-education-resources-materials/sodium-your-diet​
* https://www.sciencedirect.com/science/article/pii/S0013935120305703#sec3​
* https://www.sciencedirect.com/science/article/abs/pii/S0304389421007421?via%3Dihub​
* https://www.un.org/development/desa/dpad/wp-content/uploads/sites/45/WESP2022_ANNEX.pdf ​
* https://ourworldindata.org/plastic-pollution​
* https://data.worldbank.org/indicator/NY.GDP.MKTP.CD​
​
---

## 📁 Project Structure
```
.
├── data_utils.py         # Contains functions for loading and merging data
├── simulation.py         # Contains functions for Monte Carlo simulation
├── visualizations.py     # Contains functions for generating plots 
                            and visual summaries of simulation results
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
   pip install -r requirements.txt
   
---

## ▶️ How to Run
Once your environment is set up and dependencies are installed, run the main script:
```bash
python3 main.py