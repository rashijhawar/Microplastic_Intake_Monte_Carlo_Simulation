import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')


def load_data(air_file: str, food_file: str, water_file: str) -> pd.DataFrame:
    """
    Load and merge microplastic intake data from different sources.

    :param air_file: Path to the air microplastic intake data file.
    :param food_file: Path to the food microplastic intake data file.
    :param water_file: Path to the water microplastic intake data file.
    :return: Combined dataset indexed by country.

    >>> load_data('data/air.csv', 'data/food.csv', 'absent_file.csv') # doctest: +ELLIPSIS
    Error loading data: ...
    """
    try:
        # Load Air Data
        mp_air_intake_data = pd.read_csv(air_file)
        mp_air_intake_data.set_index("Country", inplace=True)

        # Load Food Data
        food_data = pd.read_csv(food_file)
        food_data.set_index("Country", inplace=True)
        food_data["Food Microplastic Intake (mg/capita/day)"] = food_data.sum(axis=1)
        mp_food_intake_data = food_data[["Food Microplastic Intake (mg/capita/day)"]]

        # Load Water Data
        mp_water_intake_data = pd.read_csv(water_file)
        mp_water_intake_data.set_index("Country", inplace=True)

        # Combine Air, Food, and Water Data
        mp_intake_data = mp_air_intake_data.join(mp_food_intake_data, how='inner').join(mp_water_intake_data, how='inner')

        return mp_intake_data

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
    

def filter_data(data: pd.DataFrame, countries: list) -> pd.DataFrame:
    """
    Filter the input DataFrame to include only rows corresponding to the specified countries.

    :param data: A DataFrame with countries as the index.
    :param countries: A list of country names to filter the DataFrame on.
    :return: A DataFrame containing only the rows corresponding to the specified countries.

    >>> test_df = pd.DataFrame({'Microplastic Intake': [12, 14, 16, 18, 20]},
    ... index=['United States', 'United Kingdom', 'India', 'Indonesia', 'Mexico'])
    >>> output = filter_data(test_df, ['United States', 'United Kingdom', 'India'])
    >>> sorted(output.index.tolist())
    ['India', 'United Kingdom', 'United States']
    >>> int(output.loc['India', 'Microplastic Intake'])
    16
    """
    return data.loc[countries]


def mod_pert_random(low, likely, high, confidence=4, samples=1000):
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


def run_monte_carlo_simulation(mp_intake_data: pd.DataFrame, simulations: int = 1000, verbose: bool = True) -> pd.DataFrame:
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
            mean_food = mp_intake_data.loc[country, 'Food Microplastic Intake (mg/capita/day)']
            mean_water = mp_intake_data.loc[country, 'Water Microplastic Intake (mg/capita/day)']

            # Standard deviations (25% of mean)
            std_air = mean_air * 0.25
            std_food = mean_food * 0.25
            std_water = mean_water * 0.25

            # Lognormal distribution parameters
            lognormal_mu = lambda mean, std: np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
            lognormal_sigma = lambda mean, std: np.sqrt(np.log(1 + (std ** 2 / mean ** 2)))

            air_mu, air_sigma = lognormal_mu(mean_air, std_air), lognormal_sigma(mean_air, std_air)
            food_mu, food_sigma = lognormal_mu(mean_food, std_food), lognormal_sigma(mean_food, std_food)
            water_mu, water_sigma = lognormal_mu(mean_water, std_water), lognormal_sigma(mean_water, std_water)

            # Generate samples
            air_samples = np.random.lognormal(air_mu, air_sigma, simulations)
            food_samples = np.random.lognormal(food_mu, food_sigma, simulations)
            water_samples = np.random.lognormal(water_mu, water_sigma, simulations)

            # Generate particle weights
            particle_weights = mod_pert_random(1.4e-8, 2.2e-7, 0.014, 6, simulations)

            # Compute ingestion and inhalation
            ingestion = food_samples + water_samples
            inhalation = air_samples * particle_weights
            total_mp = ingestion + inhalation

            inhalation_pct = (inhalation / total_mp) * 100
            ingestion_pct = (ingestion / total_mp) * 100

            # Build result DataFrame
            country_results = pd.DataFrame({
                'Country': country,
                'Simulation': np.arange(1, simulations + 1),
                'Daily_MP_Air': inhalation,
                'Daily_MP_Food': food_samples,
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

    # Combine results
    final_results = pd.concat(results).reset_index(drop=True)

    # Compute annual values
    for source in ['Air', 'Food', 'Water', 'Ingestion', 'Inhalation', 'Total']:
        final_results[f'Annual_MP_{source}'] = final_results[f'Daily_MP_{source}'] * 365

    return final_results


def plot_convergence(simulation_results: pd.DataFrame) -> None:
    """
    Plot convergence.
    
    :param simulation_results: A DataFrame containing the Monte Carlo Simulation results.
    """
    plt.figure(figsize=(12, 8))

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
    plt.legend(title='Country', fontsize=10)
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
         'Inhalation_Contribution_Pct', 'Ingestion_Contribution_Pct']
    ].mean().reset_index()
    country_means.sort_values('Daily_MP_Total', ascending=False, inplace=True)

    if verbose:
        print("\nMean Microplastic Intake by Country (in mg/day):")
        print("="*80)
        print(country_means[['Country', 'Daily_MP_Inhalation', 'Daily_MP_Ingestion', 'Daily_MP_Total']].to_string(index=False))
        print("="*80)

    return country_means


def plot_total_daily_mp_intake(country_means: pd.DataFrame) -> None:
    """
    Create a bar chart showing total daily microplastic intake by country.

    :param country_means: A DataFrame containing the mean values for each country.
    """
    plt.figure(figsize=(12, 6))
    
    ax = plt.bar(country_means['Country'], country_means['Daily_MP_Total'], color='mediumaquamarine')

    for bar, value in zip(ax, country_means['Daily_MP_Total']):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 value + 0.01 * max(country_means['Daily_MP_Total']), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=9)

    plt.title('Total Daily Microplastic Intake', fontsize=14)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Total Microplastic Intake (mg/day)', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    

def plot_microplastic_source_breakdown(country_means: pd.DataFrame) -> None:
    """
    Create a stacked bar chart showing the percentage breakdown of microplastic intake sources by country.
    This function visualizes the relative contributions of inhalation (air) and
    ingestion (food + water) to total microplastic exposure across different countries.

    :param country_means: A DataFrame containing the mean values for each country.
    """
    plt.figure(figsize=(12, 6))
    
    inhalation_pct = country_means['Inhalation_Contribution_Pct']
    ingestion_pct = country_means['Ingestion_Contribution_Pct']
    
    plt.bar(country_means['Country'], inhalation_pct, label='Inhalation (Air)', color='skyblue')
    plt.bar(country_means['Country'], ingestion_pct, bottom=inhalation_pct, 
            label='Ingestion (Food + Water)', color='salmon')
    
    for i, (_, air_pct, ing_pct) in enumerate(zip(country_means['Country'], inhalation_pct, ingestion_pct)):
        if air_pct > 5:
            plt.text(i, air_pct/2, f'{air_pct:.1f}%', ha='center', va='center', 
                     color='black', fontsize=9)
        
        if ing_pct > 5:
            plt.text(i, air_pct + ing_pct/2, f'{ing_pct:.1f}%', 
                     ha='center', va='center', color='black', fontsize=9)
    
    plt.title('Daily Microplastic Intake Composition', fontsize=14)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Percentage Contribution', fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=90, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    

def plot_intake_over_a_year(country_means: pd.DataFrame, country: str) -> None:
    """
    Create a line plot showing cumulative microplastic intake over a year for a specific country.

    :param country_means: A DataFrame containing the mean values for each country.
    :param country: The name of the country to plot, must match a value in the 'Country' column of country_means.
    """
    daily_total_mg = country_means.loc[country_means['Country'] == country, 'Daily_MP_Total'].values[0]
    daily_total_g = daily_total_mg / 1000 

    days = list(range(1, 366))
    cumulative_intake = pd.Series([daily_total_g] * 365).cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(days, cumulative_intake, linestyle='-', color='dimgrey')
    plt.xlabel('Day of the Year')
    plt.ylabel('Cumulative Microplastic Intake (g)')
    plt.title(f'Microplastic Intake Over A Year for {country}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def load_country_indicators_data(country_means: pd.DataFrame) -> pd.DataFrame:
    """
    Load and merge country indicator data from multiple sources with microplastic data.
    This function combines GDP data, waste management metrics, development status classifications,
    and microplastic measurements into a single comprehensive dataset for analysis.

    :param country_means: A DataFrame containing the mean values for each country.
    :return: A DataFrame containing the combined indicator data.
    """
    # Read GDP data
    gdp_data = pd.read_csv("Data/World_GDP_Data.csv", skiprows=4)
    gdp_data = gdp_data[['Country Name', '2023']]
    gdp_data.rename(columns={'Country Name': 'Country', '2023': 'GDP (current US$) - 2023'}, inplace=True)
    gdp_data.set_index('Country', inplace=True)

    # Read mismanaged plastic waster per capita data
    waste_data = pd.read_csv('Data/mismanaged-plastic-waste-per-capita.csv')
    waste_data.rename(columns={'Entity': 'Country'}, inplace=True)

    # Development status of countries
    development_status = {
        'Canada': 'Developed',
        'United States': 'Developed',
        'Mexico': 'Developing',
        'Cuba': 'Developing',
        'Dominican Republic': 'Developing',
        'Dominica': 'Developing',
        'Saint Lucia': 'Developing',
        'Barbados': 'Developing',
        'Brazil': 'Developing',
        'Argentina': 'Developing',
        'Bolivia': 'Developing',
        'Paraguay': 'Developing',
        'Peru': 'Developing',
        'Colombia': 'Developing',
        'Venezuela': 'Developing',
        'Uruguay': 'Developing',
        'China': 'Developing',
        'United Kingdom': 'Developed',
        'France': 'Developed',
        'India': 'Developing',
        'Indonesia': 'Developing',
        'Mongolia': 'Developing',
        'Russia': 'Developing',
        'Australia': 'Developed',
        'Sri Lanka': 'Developing',
        'Pakistan': 'Developing',
        'Bangladesh': 'Developing',
        'Iran': 'Developing',
        'Saudi Arabia': 'Developing',
        'Iraq': 'Developing',
        'Turkey': 'Developing',
        'Sweden': 'Developed',
        'Germany': 'Developed',
        'Ireland': 'Developed',
        'Spain': 'Developed',
        'Portugal': 'Developed',
        'Switzerland': 'Developed',
        'Austria': 'Developed',
        'Slovakia': 'Developed',
        'Hungary': 'Developed',
        'Croatia': 'Developed',
        'Bosnia and Herzegovina': 'Developing',
        'Serbia': 'Developing',
        'Romania': 'Developed',
        'Ukraine': 'Developing',
        'Philippines': 'Developing',
        'Malaysia': 'Developing',
        'Vietnam': 'Developing',
        'Cambodia': 'Developing',
        'Thailand': 'Developing',
        'South Korea': 'Developing',
        'Japan': 'Developed'
    }
    development_status_df = pd.DataFrame(list(development_status.items()), columns=['Country', 'Development Status'])

    merged_data = pd.merge(development_status_df, gdp_data, on='Country', how='inner')
    merged_data = pd.merge(merged_data, waste_data, on='Country', how='inner')
    merged_data = pd.merge(merged_data, country_means, on='Country', how='inner')

    merged_data['GDP (Million US$)'] = merged_data['GDP (current US$) - 2023'] / 1000000

    return merged_data


def plot_gdp_vs_mp_intake(country_data: pd.DataFrame) -> None:
    """
    Create a scatter plot to visualize the correlation between countries' GDP and microplastic intake.

    :param country_data: A DataFrame containing the combined indicator data.
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=country_data, x='GDP (Million US$)', y='Daily_MP_Total', 
                    hue='Development Status', palette={'Developed': 'limegreen', 'Developing': 'gold'})
    sns.regplot(data=country_data, x='GDP (Million US$)', y='Daily_MP_Total', scatter=False, color='dimgrey')
    
    plt.title('Correlation between GDP and Microplastic Intake')
    plt.xlabel('GDP (Million US$)')
    plt.ylabel('Microplastic Intake (mg/day)')
    plt.ticklabel_format(style='plain', axis='x')
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_mismanaged_waste_vs_mp_intake(country_data: pd.DataFrame) -> None:
    """
    Create a scatter plot to visualize the correlation between mismanaged plastic waste per capita and microplastic intake.

    :param country_data: A DataFrame containing the combined indicator data.
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=country_data, x='Mismanaged plastic waste per capita (kg per year)', y='Daily_MP_Total', 
                    hue='Development Status', palette={'Developed': 'limegreen', 'Developing': 'gold'})
    sns.regplot(data=country_data, x='Mismanaged plastic waste per capita (kg per year)', y='Daily_MP_Total', scatter=False, color='dimgrey')
        
    plt.title('Correlation between Mismanaged Plastic Waste per Capita and Microplastic Intake')
    plt.xlabel('Mismanaged plastic waste per capita (kg/year)')
    plt.ylabel('Microplastic Intake (mg/day)')
    plt.ticklabel_format(style='plain', axis='x')
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the microplastic intake simulation and analysis.
    """
    # Load data
    print("Loading microplastic intake data...")
    all_mp_intake_data = load_data('Data/Air_Microplastic_Intake_Data.csv',
                                   'Data/Food_Microplastic_Intake_Data.csv',
                                   'Data/Water_Microplastic_Intake_Data.csv')    
    print(f"Data loaded successfully.")

    filter_countries = ['United States', 'India', 'China', 'Malaysia', 'Indonesia', 'Japan', 
                        'Thailand', 'France', 'United Kingdom', 'Saudi Arabia', 'Brazil', 'Cuba']
    filtered_mp_intake_data = filter_data(all_mp_intake_data, filter_countries)
    
    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
    simulation_results = run_monte_carlo_simulation(filtered_mp_intake_data, simulations=10000)
    print(f"Simulation complete!")
    
    # Generate convergence plots
    print("\nGenerating convergence plot...")
    plot_convergence(simulation_results)

    # Calculate country means
    country_means = calculate_country_means(simulation_results)
    
    # Plot visualizations
    plot_total_daily_mp_intake(country_means)
    plot_microplastic_source_breakdown(country_means)
    plot_intake_over_a_year(country_means, 'United States')

    # Load country indicators data
    country_data = load_country_indicators_data(country_means)

    # TODO: Run the simulation for all countries and call these functions for all countries
    plot_gdp_vs_mp_intake(country_data)
    plot_mismanaged_waste_vs_mp_intake(country_data)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()