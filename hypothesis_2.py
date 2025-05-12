import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define dietary groups globally for reuse
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


def run_monte_carlo_simulation(file_path, n_simulations=2000, std_fraction=0.25):
    """
    Run Monte Carlo simulations to estimate microplastic (MP) intake for different diet groups across countries.

    Parameters:
    - file_path (str): Path to the Excel file containing MP intake means.
    - n_simulations (int): Number of simulation iterations.
    - std_fraction (float): Fraction of the mean to use as standard deviation.

    Returns:
    - sim_df (DataFrame): Long-form dataframe of simulation results by country and diet group.
    - food_sim_df (DataFrame): Wide-form dataframe of mean MP intake per food item per country.

    Example:
    >>> test_sim_df, test_food_df = run_monte_carlo_simulation('Data/test_food_mp_intake_data.xlsx')
    >>> test_sim_df.columns
    Index(['Country', 'DietGroup', 'MP_Intake'], dtype='object')
    >>> isinstance(test_food_df, pd.DataFrame)
    True
    >>> test_food_df.shape[1]
    18
    """
    food_intake = pd.read_excel(file_path, sheet_name='food_intake')
    food_intake['Country']=food_intake['Country'].str.strip()
    food_intake_df = food_intake.set_index('Country')

    mp_concentration = pd.read_excel(file_path, sheet_name='mp_concentration')
    mp_concentration['Country'] = mp_concentration['Country'].str.strip()
    mp_concentration_df = mp_concentration.set_index('Country')

    results = []
    food_sim_results = []

    for country in food_intake_df.index:
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

    sim_df = pd.DataFrame(results)
    food_sim_df = pd.DataFrame(food_sim_results).set_index('Country')

    return sim_df, food_sim_df


def plot_violin(sim_df, top=True, n=10, show=True):
    """
    Plot violin plots for top or least n countries by MP intake.

    Paramaters:
    - sim_df (pd.DataFrame): Simulation data with 'Country', 'DietGroup', and 'MP_Intake' columns.
    - top (bool): If True, plot top n countries; otherwise bottom n.
    - n (int): Number of countries to plot.
    - show (bool): If True, display the plot. Set to False for testing.

    Returns
    matplotlib.axes._axes.Axes: Axes object of the plot.

    >>> test_sim_df, test_food_df = run_monte_carlo_simulation('Data/test_food_mp_intake_data.xlsx')
    >>> test_ax = plot_violin(test_sim_df, True, 2, False)
    >>> isinstance(test_ax, plt.Axes)
    True
    >>> len(test_ax.get_xticklabels())
    2


    """
    avg_intake = sim_df.groupby(['Country', 'DietGroup'])['MP_Intake'].mean().reset_index()
    ranked_countries = avg_intake.groupby('Country')['MP_Intake'].mean()
    target_countries = ranked_countries.nlargest(n).index.tolist() if top else ranked_countries.nsmallest(
        n).index.tolist()

    subset_df = sim_df[sim_df['Country'].isin(target_countries)]

    plt.figure(figsize=(20, 10))
    ax = sns.violinplot(data=subset_df, x='Country', y='MP_Intake', hue='DietGroup', split=False, palette='Set2')
    title_prefix = 'Top' if top else 'Bottom'
    plt.title(f'Monte Carlo Simulation: MP Intake by Diet Type for {title_prefix} {n} Countries')
    plt.xticks(rotation=45)
    plt.ylabel("Simulated MP Intake (mg/day)")
    plt.legend(title='Diet Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if show:
        plt.show()

    return ax


def plot_stacked_bar(food_sim_df, top=True, n=10, show=True):
    """
    Plot stacked bar chart of food items contributing to MP intake.

    Parameters:
    - food_sim_df (pd.DataFrame): DataFrame with countries as index and food items as columns.
    - top (bool): If True, plot top n countries; otherwise bottom n.
    - n (int): Number of countries to plot.
    - show (bool): If True, display the plot. Set to False for doctest or testing.

    Returns:
    matplotlib.axes._axes.Axes: Axes object of the plot.

    Example
    -------
    >>> test_sim_df, test_food_df = run_monte_carlo_simulation('Data/test_food_mp_intake_data.xlsx')
    >>> test_ax = plot_stacked_bar(test_food_df, top=True, n=2, show=False)
    >>> isinstance(test_ax, plt.Axes)
    True
    >>> len(test_ax.patches) > 0
    True
    >>> len(test_ax.containers) == 18  #for 18 different food items
    True

    """
    food_sim_df['Total_Intake'] = food_sim_df.sum(axis=1)
    target_countries = food_sim_df['Total_Intake'].nlargest(n).index.tolist() if top else food_sim_df[
        'Total_Intake'].nsmallest(n).index.tolist()

    food_df = food_sim_df.loc[target_countries].drop(columns='Total_Intake')
    ax = food_df.plot(kind='bar', stacked=True, figsize=(20, 10), colormap='tab20')
    title_prefix = 'Top' if top else 'Bottom'
    plt.title(f'Contribution of Food Items to MP Intake ({title_prefix} {n} Countries)')
    plt.ylabel('Simulated MP Intake (mg/day)')
    plt.xlabel('Country')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Food Item', ncol=2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if show:
        plt.show()

    return ax