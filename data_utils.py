import pandas as pd


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
        food_intake_data = pd.read_excel(food_file, sheet_name='food_intake')
        food_intake_data['Country'] = food_intake_data['Country'].str.strip()
        food_intake_data.set_index('Country', inplace=True)

        food_mp_conc_data = pd.read_excel(food_file, sheet_name='mp_concentration')
        food_mp_conc_data['Country'] = food_mp_conc_data['Country'].str.strip()
        food_mp_conc_data.set_index('Country', inplace=True)

        # Load Water Data
        mp_water_intake_data = pd.read_csv(water_file)
        mp_water_intake_data.set_index("Country", inplace=True)

        # Combine Air, Food, and Water Data
        mp_air_water_intake_data = mp_air_intake_data.join(mp_water_intake_data, how='inner')

        return mp_air_water_intake_data, food_intake_data, food_mp_conc_data

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


def load_country_indicators_data(country_means: pd.DataFrame) -> pd.DataFrame:
    """
    Load and merge country indicator data from multiple sources with microplastic data.

    :param country_means: A DataFrame containing the mean values for each country.
    :return: A DataFrame containing the combined indicator data.
    """
    # Read GDP data
    try:
        gdp_data = pd.read_csv("Data/World_GDP_Data.csv", skiprows=4)
        gdp_data = gdp_data[['Country Name', '2023']]
        gdp_data.rename(columns={'Country Name': 'Country', '2023': 'GDP (current US$) - 2023'}, inplace=True)
        gdp_data.set_index('Country', inplace=False, drop=False)
    except FileNotFoundError:
        raise FileNotFoundError("GDP data file not found. Please ensure 'Data/World_GDP_Data.csv' exists.")
    except KeyError:
        raise KeyError("Expected columns 'Country Name' or '2023' not found in GDP data.")

    # Read mismanaged plastic waste per capita data
    try:
        waste_data = pd.read_csv('Data/mismanaged-plastic-waste-per-capita.csv')
        if 'Entity' not in waste_data.columns:
            raise KeyError("Column 'Entity' not found in waste data.")
        waste_data.rename(columns={'Entity': 'Country'}, inplace=True)
    except FileNotFoundError:
        raise FileNotFoundError("Waste data file not found. Please ensure 'Data/mismanaged-plastic-waste-per-capita.csv' exists.")

    # Development status mapping
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

    try:
        merged_data = pd.merge(development_status_df, gdp_data, on='Country', how='inner')
        merged_data = pd.merge(merged_data, waste_data, on='Country', how='inner')
        merged_data = pd.merge(merged_data, country_means, on='Country', how='inner')
    except Exception as e:
        raise RuntimeError(f"Error while merging data: {e}")

    if 'GDP (current US$) - 2023' not in merged_data.columns:
        raise KeyError("GDP column missing after merge.")
    merged_data['GDP (Trillion US$)'] = merged_data['GDP (current US$) - 2023'] / 1_000_000_000_000

    return merged_data
