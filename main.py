from data_utils import *
from simulation import *
from visualizations import *


def main():
    """
    Main function to run the microplastic intake simulation and analysis.
    """
    # Load data
    try:
        print("Loading microplastic intake data...")
        mp_air_water_intake_data, food_intake_data, food_mp_conc_data = load_data(
            'Data/Air_Microplastic_Intake_Data.csv',
            'Data/Food_Microplastic_Intake_Data.xlsx',
            'Data/Water_Microplastic_Intake_Data.csv'
        )
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Validation
    try:
        filter_countries = ['United States', 'Indonesia', 'Paraguay']
        filtered_mp_air_water_intake_data = filter_data(mp_air_water_intake_data, filter_countries)
        filtered_food_intake_data = filter_data(food_intake_data, filter_countries)
        filtered_food_mp_conc_data = filter_data(food_mp_conc_data, filter_countries)

        print("\nRunning Monte Carlo simulation...")
        simulation_results = run_monte_carlo_simulation(
            filtered_mp_air_water_intake_data,
            filtered_food_intake_data,
            filtered_food_mp_conc_data,
            simulations=2000
        )
        print("Simulation complete!")
    except Exception as e:
        print(f"Error during Monte Carlo simulation (initial): {e}")
        return

    try:
        country_means = calculate_country_means(simulation_results)
    except Exception as e:
        print(f"Error calculating country means: {e}")
        return

    # Hypothesis 1
    try:
        # Run Monte Carlo simulation
        print("\nRunning Monte Carlo simulation for Hypothesis 1...")
        simulation_results = run_monte_carlo_simulation(
            mp_air_water_intake_data,
            food_intake_data,
            food_mp_conc_data,
            simulations=2000
        )
        print("Simulation complete!")

        # Calculate country means
        country_means = calculate_country_means(simulation_results)
        filtered_country_means = country_means.head(15) if len(country_means) > 15 else country_means

        # Generate convergence plot
        print("\nGenerating convergence plot...")
        plot_convergence(simulation_results, filtered_country_means['Country'].unique())

        # Plot visualizations 
        plot_total_daily_mp_intake(filtered_country_means)
        plot_total_annual_mp_intake(filtered_country_means)
        plot_microplastic_source_breakdown(filtered_country_means)
    except Exception as e:
        print(f"Error during Hypothesis 1 analysis: {e}")
        return

    try:
        # Load country indicators data
        print("\nLoading country indicators data...")
        country_data = load_country_indicators_data(country_means)
        print("Data loaded successfully.")

        # Plot visualizations for developed and developing countries
        plot_mean_daily_mp_intake_for_developed_and_developing_countries(country_data)
        plot_daily_mp_intake_by_development_status(country_data)

       # We notice that China is an outlier. Let's reduce the impact of the outlier by replacing China's intake with the mean.
        country_data_copy = country_data.copy()
        mean_intake = country_data_copy[
            (country_data_copy['Development Status'] == 'Developing') &
            (country_data_copy['Country'] != 'China')
        ]['Daily_MP_Total'].mean()
        country_data_copy.loc[country_data_copy['Country'] == 'China', 'Daily_MP_Total'] = mean_intake
        plot_mean_daily_mp_intake_for_developed_and_developing_countries(country_data_copy)

        # Plot visualizations with country indicators
        plot_gdp_vs_mp_intake(country_data)
        plot_mismanaged_waste_vs_mp_intake(country_data)

        # Plot total microplastic intake on the world map
        #plot_total_mp_intake_on_map(country_means)
    except Exception as e:
        print(f"Error during country indicators analysis: {e}")
        return

    # Hypothesis 2
    try:
        # Run Monte Carlo simulation
        print("\nRunning Monte Carlo simulation for Hypothesis 2...")
        sim_df, food_df = run_monte_carlo_simulation_hypothesis2('Data/Food_Microplastic_Intake_Data.xlsx')
        print("Simulation complete!")

        # Plot visualizations
        plot_violin(sim_df, True, 10)
        plot_violin(sim_df, False, 10)
        plot_stacked_bar(food_df, True, 10)
        plot_stacked_bar(food_df, False, 10)
    except Exception as e:
        print(f"Error during Hypothesis 2 analysis: {e}")
        return

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()