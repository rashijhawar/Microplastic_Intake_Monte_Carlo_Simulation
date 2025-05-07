from data_utils import *
from simulation import *
from visualizations import *

def main():
    """
    Main function to run the microplastic intake simulation and analysis.
    """
    # Load data
    print("Loading microplastic intake data...")
    mp_air_water_intake_data, food_intake_data, food_mp_conc_data = load_data(
        'Data/Air_Microplastic_Intake_Data.csv',
        'Data/Food_Microplastic_Intake_Data.xlsx',
        'Data/Water_Microplastic_Intake_Data.csv'
    )    
    print("Data loaded successfully.")

    # -------------------------------------------------------------------------------------------
    # Validation
    filter_countries = ['United States', 'Indonesia', 'Paraguay']
    filtered_mp_air_water_intake_data = filter_data(mp_air_water_intake_data, filter_countries)
    filtered_food_intake_data = filter_data(food_intake_data, filter_countries)
    filtered_food_mp_conc_data = filter_data(food_mp_conc_data, filter_countries)

    print("\nRunning Monte Carlo simulation...")
    simulation_results = run_monte_carlo_simulation(filtered_mp_air_water_intake_data, filtered_food_intake_data, filtered_food_mp_conc_data, simulations=5000)
    print("Simulation complete!")

    country_means = calculate_country_means(simulation_results)
    
    # -------------------------------------------------------------------------------------------
    # Hypothesis 1
    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
    simulation_results = run_monte_carlo_simulation(mp_air_water_intake_data, food_intake_data, food_mp_conc_data, simulations=5000)
    print("Simulation complete!")

    # Calculate country means
    country_means = calculate_country_means(simulation_results)
    if len(country_means) > 15:
        filtered_country_means = country_means.head(15)
    
    # Generate convergence plot
    print("\nGenerating convergence plot...")
    plot_convergence(simulation_results, filtered_country_means['Country'].unique())
    
    # Plot visualizations
    plot_total_daily_mp_intake(filtered_country_means)
    plot_microplastic_source_breakdown(filtered_country_means)

    # Load country indicators data
    print("Loading country indicators data...")
    country_data = load_country_indicators_data(country_means)
    print("Data loaded successfully.")

    # Plot visualizations with country indicators
    plot_gdp_vs_mp_intake(country_data)
    plot_mismanaged_waste_vs_mp_intake(country_data)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()