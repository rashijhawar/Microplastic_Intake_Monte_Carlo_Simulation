import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')


def mod_pert_random(low: float, likely: float, high: float, confidence: int = 4, samples: int = 1000) -> np.ndarray:
    """
    Produce random numbers according to the Modified PERT distribution.
    Note: This function has been taken from Professor's Probability_Distributions.ipynb.
    
    :param low: The lowest value expected as possible.
    :param likely: The 'most likely' value, statistically, the mode.
    :param high: The highest value expected as possible.
    :param confidence: This is typically called 'lambda' in literature about the Modified PERT distribution. 
                       The value 4 here matches the standard PERT curve. 
                       Higher values indicate higher confidence in the mode.
                       Currently, allows values 1-18.
    :param samples: Number of random samples to generate from the distribution. (Default: 1000)
    :return: A numpy array containing random numbers following the Modified PERT distribution.

    >>> output = mod_pert_random(10, 20, 30, samples=5)
    >>> len(output)
    5
    >>> all(10 <= i <= 30 for i in output)
    True
    >>> output = mod_pert_random(10, 20, 30, confidence=19)
    Traceback (most recent call last):
    ValueError: confidence value must be in range 1-18.
    """
    if confidence < 1 or confidence > 18:
        raise ValueError('confidence value must be in range 1-18.')
        
    mean = (low + confidence * likely + high) / (confidence + 2)

    a = (mean - low) / (high - low) * (confidence + 2)
    b = ((confidence + 1) * high - low - confidence * likely) / (high - low)
    
    beta = np.random.beta(a, b, samples)
    beta = beta * (high - low) + low
    return beta


def generate_lognormal_samples(mean: float, simulations: int = 1000) -> np.ndarray:
    """
    Generate samples from a lognormal distribution.
    
    :param mean: Mean value.
    :param simulations: Number of simulations to generate.
    :return: Array of samples from the lognormal distribution.
    """
    if mean == 0:
        return np.zeros(simulations)
    
    std = mean * 0.25
    
    # Lognormal distribution parameters
    lognormal_mu = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
    lognormal_sigma = np.sqrt(np.log(1 + (std ** 2 / mean ** 2)))
    
    return np.random.lognormal(lognormal_mu, lognormal_sigma, simulations)


def run_monte_carlo_simulation(mp_intake_data: pd.DataFrame, food_intake_data: pd.DataFrame, food_mp_conc_data: pd.DataFrame, simulations: int = 1000, verbose: bool = True) -> pd.DataFrame:
    """
    Run a Monte Carlo simulation of daily microplastic intake (in mg) per person for countries present in mp_intake_data.

    :param mp_intake_data: Microplastic intake data indexed by country.
    :param simulations: Number of simulations per country. (Default: 1000)
    :param verbose: A boolean flag to enable printed output.
    :return: A DataFrame containing the simulation results.

    >>> test_df = pd.DataFrame({
    ... 'Air Microplastic Intake (particles/capita/day)': [1000, 2000, 3000, 4000, 5000],
    ... 'Food Microplastic Intake (mg/capita/day)': [5, 10, 15, 20, 25],
    ... 'Water Microplastic Intake (mg/capita/day)': [3, 6, 9, 12, 15]
    ... }, index=['United States', 'United Kingdom', 'India', 'Indonesia', 'Mexico'])
    >>> output = run_monte_carlo_simulation(test_df, simulations=10, verbose=False)
    >>> len(output) == 50
    True
    """
    results = []

    for country in mp_intake_data.index:
        try:
            if verbose:
                print(f"Running simulation for {country}...")

            # Get country-specific means
            mean_air = mp_intake_data.loc[country, 'Air Microplastic Intake (particles/capita/day)']
            mean_water = mp_intake_data.loc[country, 'Water Microplastic Intake (mg/capita/day)']

            # Generate samples for air and water
            air_samples = generate_lognormal_samples(mean_air, simulations=simulations)
            water_samples = generate_lognormal_samples(mean_water, simulations=simulations)

            # Generate samples for particle weights
            particle_weights = mod_pert_random(1.4e-8, 2.2e-7, 0.014, 6, simulations)

            # Initialize food microplastic intake array
            food_mp_intake = np.zeros(simulations)
            
            # For each food category, calculate microplastic intake
            for category in food_intake_data.columns:
                mean_intake = food_intake_data.loc[country, category]
                mean_conc = food_mp_conc_data.loc[country, category]
                
                # Generate samples for intake and concentration
                intake_samples = generate_lognormal_samples(mean_intake, simulations=simulations)
                conc_samples = generate_lognormal_samples(mean_conc, simulations=simulations)
                
                # Calculate microplastic intake for this food category
                category_mp_intake = intake_samples * conc_samples
                
                # Add to total food microplastic intake
                food_mp_intake += category_mp_intake

            # Compute ingestion and inhalation
            ingestion = food_mp_intake + water_samples
            inhalation = air_samples * particle_weights
            total_mp = ingestion + inhalation

            inhalation_pct = (inhalation / total_mp) * 100
            ingestion_pct = (ingestion / total_mp) * 100

            # Store results
            country_results = pd.DataFrame({
                'Country': country,
                'Simulation': np.arange(1, simulations + 1),
                'Daily_MP_Air': inhalation,
                'Daily_MP_Food': food_mp_intake,
                'Daily_MP_Water': water_samples,
                'Daily_MP_Ingestion': ingestion,
                'Daily_MP_Inhalation': inhalation,
                'Daily_MP_Total': total_mp,
                'Inhalation_Contribution_Pct': inhalation_pct,
                'Ingestion_Contribution_Pct': ingestion_pct
            })

            results.append(country_results)

        except Exception as e:
            print(f"Error processing country {country}: {e}")

    final_results = pd.concat(results).reset_index(drop=True)

    final_results['Monthly_MP_Total (in grams)'] = final_results['Daily_MP_Total'] * 30 / 1000
    final_results['Monthly_MP_Ingestion (in grams)'] = final_results['Daily_MP_Ingestion'] * 30 / 1000
    final_results['Annual_MP_Total (in grams)'] = final_results['Daily_MP_Total'] * 365 / 1000

    return final_results


def plot_convergence(simulation_results: pd.DataFrame, country_list: list = None) -> None:
    """
    Plot convergence.
    
    :param simulation_results: A DataFrame containing the Monte Carlo Simulation results.
    :param country_list: List of countries to display in the convergence plot.
    """
    plt.figure(figsize=(12, 8))

    if country_list is None:
        country_list = simulation_results['Country'].unique()

    if len(country_list) > 10:
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
    else:
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)

    for country in country_list:
        country_df = simulation_results[simulation_results['Country'] == country].sort_values('Simulation')
        iterations = country_df['Simulation'].tolist()
        values = country_df['Daily_MP_Total'].tolist()

        running_mean = np.cumsum(values) / np.arange(1, len(values) + 1)
        plt.plot(iterations, running_mean, label=country)

    plt.title(f'Convergence Plot', fontsize=14)
    plt.xlabel('Number of Simulations', fontsize=12)
    plt.ylabel('Total Microplastic Intake (mg/day)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Country', fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def calculate_country_means(simulation_results: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Calculates mean values for each country from the simulation results.

    :param simulation_results: A DataFrame containing the simulation results.
    :param verbose: A boolean flag to enable printed output.
    :return: A DataFrame containing the mean values for each country.

    >>> test_df = pd.DataFrame({
    ... 'Country': ['United States', 'United States', 'Indonesia', 'Indonesia'],
    ... 'Daily_MP_Inhalation': [5, 4, 10, 11],
    ... 'Daily_MP_Ingestion': [15, 12, 30, 33],
    ... 'Daily_MP_Total': [20, 16, 40, 44],
    ... 'Inhalation_Contribution_Pct': [25, 25, 25, 25],
    ... 'Ingestion_Contribution_Pct': [75, 75, 75, 75]
    ... })
    >>> output = calculate_country_means(test_df, verbose=False)
    >>> output.shape[0] == 2 and output.shape[1] == 6
    True
    >>> output.iloc[0]['Country']
    'Indonesia'
    >>> float(output.iloc[0]['Daily_MP_Total'])
    42.0
    """
    country_means = simulation_results.groupby('Country')[
        ['Daily_MP_Inhalation', 'Daily_MP_Ingestion', 'Daily_MP_Total', 
         'Monthly_MP_Total (in grams)', 'Monthly_MP_Ingestion (in grams)',
         'Inhalation_Contribution_Pct', 'Ingestion_Contribution_Pct', 'Annual_MP_Total (in grams)']
    ].mean().reset_index()
    country_means.sort_values('Daily_MP_Total', ascending=False, inplace=True)

    if verbose:
        print("\nMean Microplastic Intake by Country (in mg/day):")
        print("="*140)
        print(country_means[['Country', 'Daily_MP_Inhalation', 'Daily_MP_Ingestion', 'Daily_MP_Total', 'Monthly_MP_Total (in grams)', 'Monthly_MP_Ingestion (in grams)']].to_string(index=False))
        print("="*140)

    return country_means


def run_monte_carlo_simulation_hypothesis2(file_path: str, n_simulations: int = 2000, std_fraction: float = 0.25):
    """
    Run Monte Carlo simulations to estimate microplastic intake for different diet groups across countries.

    :param file_path: Path to the Excel file containing microplastic intake means.
    :param n_simulations: Number of simulation iterations.
    :param std_fraction: Fraction of the mean to use as standard deviation.
    :return sim_df: A dataframe of simulation results by country and diet group.
    :return A dataframe of mean microplastic intake per food item per country.

    >>> test_sim_df, test_food_df = run_monte_carlo_simulation('Data/test_food_mp_intake_data.xlsx')
    >>> test_sim_df.columns
    Index(['Country', 'DietGroup', 'MP_Intake'], dtype='object')
    >>> isinstance(test_food_df, pd.DataFrame)
    True
    >>> test_food_df.shape[1]
    18
    """
    diet_groups = {
        'omnivore': ['Cheese', 'Yogurt', 'Total Milk', 'Fruits', 'Refined Grains', 'Whole Grains', 'Nuts And Seeds',
                    'Total Processed Meats', 'Unprocessed Red Meats', 'Fish', 'Shellfish', 'Eggs', 'Total Salt',
                    'Added Sugars', 'Non-Starchy Vegetables', 'Potatoes', 'Other Starchy Vegetables',
                    'Beans And Legumes'],
        'pescetarian': ['Fruits', 'Non-Starchy Vegetables', 'Potatoes', 'Other Starchy Vegetables', 'Nuts And Seeds',
                        'Beans And Legumes', 'Refined Grains', 'Whole Grains', 'Added Sugars',
                        'Fish', 'Shellfish', 'Total Milk', 'Cheese', 'Yogurt', 'Eggs'],
        'vegetarian': ['Fruits', 'Non-Starchy Vegetables', 'Potatoes', 'Other Starchy Vegetables', 'Nuts And Seeds',
                    'Beans And Legumes', 'Refined Grains', 'Whole Grains', 'Added Sugars',
                    'Total Milk', 'Cheese', 'Yogurt', 'Eggs'],
        'vegan': ['Fruits', 'Non-Starchy Vegetables', 'Potatoes', 'Other Starchy Vegetables', 'Nuts And Seeds',
                'Beans And Legumes', 'Refined Grains', 'Whole Grains', 'Added Sugars', 'Eggs']
    }
        
    food_intake = pd.read_excel(file_path, sheet_name='food_intake')
    food_intake['Country']=food_intake['Country'].str.strip()
    food_intake_df = food_intake.set_index('Country')

    mp_concentration = pd.read_excel(file_path, sheet_name='mp_concentration')
    mp_concentration['Country'] = mp_concentration['Country'].str.strip()
    mp_concentration_df = mp_concentration.set_index('Country')

    results = []
    food_sim_results = []

    for country in food_intake_df.index:
        try:
            food_item_sums = {}
            for diet_name, food_items in diet_groups.items():
                # valid_items = [item for item in food_items if item in food_intake_df.index and not pd.isna(food_intake_df[item])]
                sims = np.zeros(n_simulations)

                for food in food_items:
                    mean_intake = food_intake_df.loc[country, food]
                    std = std_fraction * mean_intake
                    if mean_intake > 0:
                        sigma = np.sqrt(np.log(1 + (std / mean_intake) ** 2))
                        mu = np.log(mean_intake) - 0.5 * sigma ** 2
                        samples_intake = np.random.lognormal(mean=mu, sigma=sigma, size=n_simulations)
                    else:
                        samples_intake = np.zeros(n_simulations)

                    mean_conc = mp_concentration_df.loc[country, food]
                    std = std_fraction * mean_conc
                    if mean_conc > 0:
                        sigma = np.sqrt(np.log(1 + (std / mean_conc) ** 2))
                        mu = np.log(mean_conc) - 0.5 * sigma ** 2
                        samples_conc = np.random.lognormal(mean=mu, sigma=sigma, size=n_simulations)
                    else:
                        samples_conc = np.zeros(n_simulations)
                    samples_mp_intake = samples_intake * samples_conc
                    food_item_sums[food] = samples_mp_intake.mean()
                    sims += samples_mp_intake

                for val in sims:
                    results.append({
                        'Country': country,
                        'DietGroup': diet_name,
                        'MP_Intake': val
                    })
            food_sim_results.append({'Country': country, **food_item_sums})
        except Exception as e:
            print(f"Error processing country {country}: {e}")

    sim_df = pd.DataFrame(results)
    food_sim_df = pd.DataFrame(food_sim_results).set_index('Country')

    return sim_df, food_sim_df