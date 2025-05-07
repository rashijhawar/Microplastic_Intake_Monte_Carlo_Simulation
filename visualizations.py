import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

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