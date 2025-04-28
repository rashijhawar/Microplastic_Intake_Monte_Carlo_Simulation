import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def convert_to_particles_per_L(row):
    val = row['Concentration']
    unit = row['Concentration_Units']

    if unit == 'particles/L':
        return val
    elif unit == 'particles/0.33 L':
        return val * (1 / 0.33)  # ~3.03
    elif unit == 'particles/m3':
        return val / 1000  # 1000 L in 1 m³
    elif unit == 'particles/50 L':
        return val / 50
    elif unit == 'particles/mL':
        return val * 1000
    elif unit == 'particles/bottle':
        assumed_bottle_vol_L = 0.5  #CHANGE this if needed
        return val / assumed_bottle_vol_L
    else:
        return np.nan  # skip ug/m3, ug/g, nan, or unknown units

dtype_spec = {
    'Sample_ID': str,
    'Location': str,
    'Countries': str,
    'Source': str,
    'Concentration': str,
    'Concentration_Units': str,
    'Approximate_Latitude': float,
    'Approximate_Longitude': float
}
water_data = pd.read_csv('samples_geocoded.csv',
                         usecols=['Sample_ID','Location','Countries','Source','Concentration','Concentration_Units','Approximate_Latitude','Approximate_Longitude'],
                         dtype = dtype_spec,
                         na_values=['', 'NA', 'NaN', 'nan', 'null', 'NULL', '-', '?'])


food_data = pd.read_excel('Food_MP_intake_data.xlsx',sheet_name='Sheet1')
food_data.set_index('Country', inplace = True)

# Convert Concentration column to numeric, force errors to NaN
water_data['Concentration'] = pd.to_numeric(water_data['Concentration'], errors='coerce')

water_data = water_data.dropna(subset=['Concentration', 'Concentration_Units'])

water_data['Concentration_particles_per_L'] = water_data.apply(convert_to_particles_per_L, axis=1)

air_data = pd.read_csv('Air_MP_Intake_Data.csv')
air_data.set_index('Country', inplace=True)

air_breathed_in_daily = 10 # represents average amount of air breathed in by an average person daily (in cubic meters)

# Optional: View country-level averages
country_avg = water_data.groupby('Countries')['Concentration_particles_per_L'].mean().reset_index()

results = []

for country in food_data.index:

    # Example: Simulate for 1000 people in each country
    simulations_per_country = 1000
    daily_water_intake_L = np.random.normal(loc=2, scale=0.5, size=simulations_per_country)  # 2L ± 0.5
    # Choose a country — say India

    # Get average microplastic concentration for that country
    mean_concentration = country_avg.loc[country_avg['Countries'] == country, 'Concentration_particles_per_L'].values[0]

    daily_air_intake = np.random.normal(
        loc=air_data.loc[country, 'Daily MP Inhaled (particles/capita/day)'], 
        scale=air_data.loc[country, 'Daily MP Inhaled (particles/capita/day)'] * 0.25,
        size=simulations_per_country
    )

    # Simulate daily intake
    simulated_microplastic_intake_air = air_breathed_in_daily * daily_air_intake
    simulated_microplastic_intake_water = daily_water_intake_L * mean_concentration  # particles/day
    simulated_microplastic_intake_food = np.random.normal(loc=food_data.loc[country, 'Daily MP Dietary Intake(mg/capita/day)'],
                                                          scale=food_data.loc[country, 'Daily MP Dietary Intake(mg/capita/day)'] * 0.25,
                                                          size=simulations_per_country)

    # Annual exposure
    total_daily_microplastics = simulated_microplastic_intake_water + simulated_microplastic_intake_food + simulated_microplastic_intake_air
    annual_microplastic_intake = total_daily_microplastics * 365  # particles/year

    # Store
    country_df = pd.DataFrame({
        'Country': country,
        'Annual_MP_Intake': annual_microplastic_intake
    })
    results.append(country_df)

# Combine
final_results = pd.concat(results)

# Analyze
print(final_results.groupby('Country')['Annual_MP_Intake'].describe())

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(data=final_results, x='Annual_MP_Intake', hue='Country', kde=True, bins=30)
plt.title('Estimated Annual Microplastic Intake by Country')
plt.xlabel('Annual Microplastic Intake (particles)')
plt.ylabel('Density')
plt.show()
