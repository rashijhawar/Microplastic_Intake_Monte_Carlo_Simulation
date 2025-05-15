from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from geopy.geocoders import Nominatim
import time
import plotly.express as px
from simulation import run_monte_carlo_simulation_hypothesis2

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')


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

    plt.title('Total Daily Microplastic Intake Per Capita (Top 15 Countries)', fontsize=14)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Total Microplastic Intake (mg/day)', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_total_annual_mp_intake(country_means: pd.DataFrame) -> None:
    """
    Create a bar chart showing total annual microplastic intake by country.

    :param country_means: A DataFrame containing the mean values for each country.
    """
    plt.figure(figsize=(12, 6))
    
    ax = plt.bar(country_means['Country'], country_means['Annual_MP_Total (in grams)'], color='mediumaquamarine')

    for bar, value in zip(ax, country_means['Annual_MP_Total (in grams)']):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 value + 0.01 * max(country_means['Annual_MP_Total (in grams)']), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=9)

    plt.title('Total Annual Microplastic Intake Per Capita (Top 15 Countries)', fontsize=14)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Total Microplastic Intake (g/year)', fontsize=12)
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
    
    plt.title('Daily Microplastic Intake Composition (Top 15 Countries)', fontsize=14)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Percentage Contribution', fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=90, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_total_mp_intake_on_map(country_means: pd.DataFrame) -> None:
    """
    Plot the average daily microplastic intake per capita on a world map.

    :param country_means: A DataFrame containing the mean values for each country.
    """
    geolocator = Nominatim(user_agent="geoapi")
    
    latitudes = []
    longitudes = []
    for country in country_means['Country'].tolist():
        try:
            location = geolocator.geocode(country, timeout=10)
            latitudes.append(location.latitude)
            longitudes.append(location.longitude)
        except:
            latitudes.append(None)
            longitudes.append(None)
        time.sleep(1)

    country_means['Latitude'] = latitudes
    country_means['Longitude'] = longitudes

    fig = px.choropleth(
        country_means,
        locations="Country",
        locationmode="country names",
        color="Daily_MP_Total",
        color_continuous_scale="Reds",
        title="Global Microplastic Intake per Capita (mg/day)",
    )

    fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
    fig.show()


def plot_gdp_vs_mp_intake(country_data: pd.DataFrame) -> None:
    """
    Create a scatter plot to visualize the correlation between countries' GDP and microplastic intake.

    :param country_data: A DataFrame containing the combined indicator data.
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=country_data, x='GDP (Trillion US$)', y='Daily_MP_Total', 
                    hue='Development Status', palette={'Developed': 'limegreen', 'Developing': 'gold'})
    sns.regplot(data=country_data, x='GDP (Trillion US$)', y='Daily_MP_Total', scatter=False, color='dimgrey')
    
    plt.title('Correlation between GDP and Microplastic Intake')
    plt.xlabel('GDP (Trillion US$)')
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


def plot_mean_daily_mp_intake_for_developed_and_developing_countries(country_data: pd.DataFrame) -> None:
    """
    Plots a bar chart comparing the mean daily microplastic intake between developed and developing countries.

    :param country_data: A DataFrame containing the combined indicator data.
    """
    mp_intake_development_status = country_data.groupby('Development Status')['Daily_MP_Total'].mean().reset_index()

    plt.bar(mp_intake_development_status['Development Status'], mp_intake_development_status['Daily_MP_Total'], color='mediumturquoise')
    plt.ylabel('Mean Microplastic Intake (mg/day)')
    plt.xlabel('Development Status')
    plt.title('Mean Microplastic Intake by Development Status')
    plt.tight_layout()
    plt.show()


def plot_daily_mp_intake_by_development_status(country_data: pd.DataFrame) -> None:
    """
    Plots a bar chart showing daily microplastic intake for each country, 
    color-coded by development status (Developed or Developing).

    :param country_data: A DataFrame containing the combined indicator data.
    """
    df_sorted = country_data.sort_values(by='Development Status', ascending=True)
    colors = df_sorted['Development Status'].map({'Developed': 'skyblue', 'Developing': 'salmon'})

    plt.figure(figsize=(12, 6))
    plt.bar(df_sorted['Country'], df_sorted['Daily_MP_Total'], color=colors)

    plt.xlabel('Country')
    plt.ylabel('Microplastic Intake (mg/day)')
    plt.title('Microplastic Intake by Country and Development Status')
    plt.xticks(rotation=90, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    legend_elements = [Patch(facecolor='skyblue', label='Developed'), Patch(facecolor='salmon', label='Developing')]
    plt.legend(handles=legend_elements)

    plt.show()


def plot_violin(sim_df: pd.DataFrame, top: bool = True, n: int = 10, show: bool = True):
    """
    Plot violin plots for top or least n countries by MP intake.

    :param sim_df: Simulation data with 'Country', 'DietGroup', and 'MP_Intake' columns.
    :param top: If True, plot top n countries; otherwise bottom n.
    :param n: Number of countries to plot.
    :param show: If True, display the plot. Set to False for testing.

    >>> test_sim_df, test_food_df = run_monte_carlo_simulation_hypothesis2('Data/test_food_mp_intake_data.xlsx')
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

    plt.figure(figsize=(20, 8))
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


def plot_stacked_bar(food_sim_df: pd.DataFrame, top: bool = True, n: int = 10, show: bool = True):
    """
    Plot stacked bar chart of food items contributing to MP intake.

    :param food_sim_df: DataFrame with countries as index and food items as columns.
    :param top: If True, plot top n countries; otherwise bottom n.
    :param n: Number of countries to plot.
    :param show: If True, display the plot. Set to False for doctest or testing.

    >>> test_sim_df, test_food_df = run_monte_carlo_simulation_hypothesis2('Data/test_food_mp_intake_data.xlsx')
    >>> test_ax = plot_stacked_bar(test_food_df, top=True, n=2, show=False)
    >>> isinstance(test_ax, plt.Axes)
    True
    >>> len(test_ax.patches) > 0
    True
    >>> len(test_ax.containers) == 18  # for 18 different food items
    True
    """
    food_sim_df['Total_Intake'] = food_sim_df.sum(axis=1)
    target_countries = food_sim_df['Total_Intake'].nlargest(n).index.tolist() if top else food_sim_df[
        'Total_Intake'].nsmallest(n).index.tolist()

    food_df = food_sim_df.loc[target_countries].drop(columns='Total_Intake')
    ax = food_df.plot(kind='bar', stacked=True, figsize=(20, 8), colormap='tab20')
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