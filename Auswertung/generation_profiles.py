#%%
import xarray as xr
import matplotlib.pyplot as plt
import os
import pandas as pd

cwd = os.getcwd()

#%% Load the NetCDF file
file_path = os.path.join(os.path.dirname(cwd), "resources", "test", "weather_year_2015", "profile_39_solar.nc")
ds = xr.open_dataset(file_path)

#%% Display dataset metadata
print(ds)

#%% List available variables
print("\nVariables in the dataset:", list(ds.variables))
#%% Select the 'profile' variable
var_name = "profile"  # Selecting 'profile' instead of 'temperature'
if var_name in ds:
    data_var = ds[var_name]
    print(data_var)
    
    # Plot time series for a specific bus (first one in dataset)
    bus_index = 0  # Change index to select different buses
    data_var.sel(bus=ds.bus.values[bus_index]).plot(x='time')
    plt.title(f"{var_name} time series for bus {ds.bus.values[bus_index]}")
    plt.xlabel("Time")
    plt.ylabel("Profile Value")
    plt.xticks(rotation=45)
    plt.show()
else:
    print(f"Variable '{var_name}' not found in dataset. Choose from: {list(ds.variables)}")

#%% Close dataset
ds.close()

#%% Get current working directory and construct file paths for multiple years
years = range(2015, 2020)
file_paths = [os.path.join(os.path.dirname(cwd), "resources", "test", f"weather_year_{year}", "profile_39_solar.nc") for year in years]

#%% Load datasets and assign correct year values
datasets = []
for year, fp in zip(years, file_paths):
    ds = xr.open_dataset(fp)
    ds = ds.assign_coords(year=("year", [year]))  # Overwrite existing year coordinate
    datasets.append(ds)
print(datasets)
#%% Concatenate datasets along the 'year' dimension
ds = xr.concat(datasets, dim="year")
#%%
print(ds)

#%% Compute yearly average capacity factor for each bus
yearly_avg = ds["profile"].mean(dim="time")
print(yearly_avg)

#%% Convert to DataFrame and save to CSV
yearly_avg_df = yearly_avg.to_dataframe().reset_index()
yearly_avg_df = yearly_avg_df.pivot(index="year", columns="bus", values="profile")
#%% Define dictionary to map bus indices to country names

# ISO2 to Country Name Mapping
iso2_to_country = {
    'AT': 'Austria',
    'BE': 'Belgium',
    'BG': 'Bulgaria',
    'CZ': 'Czech Republic',
    'DK': 'Denmark',
    'EE': 'Estonia',
    'FI': 'Finland',
    'FR': 'France',
    'DE': 'Germany',
    'GR': 'Greece',
    'HU': 'Hungary',
    'IE': 'Ireland',
    'IT': 'Italy',
    'LV': 'Latvia',
    'LT': 'Lithuania',
    'LU': 'Luxembourg',
    'NL': 'Netherlands',
    'NO': 'Norway',
    'PL': 'Poland',
    'PT': 'Portugal',
    'RO': 'Romania',
    'SK': 'Slovakia',
    'SI': 'Slovenia',
    'ES': 'Spain',
    'SE': 'Sweden',
    'CH': 'Switzerland',
    'GB': 'United Kingdom',
    'UK': 'United Kingdom',
    'AL': 'Albania', 
    'BA': 'Bosnia and Herzegovina', 
    'HR': 'Croatia', 
    'ME': 'Montenegro', 
    'MK': 'North Macedonia', 
    'RS': 'Serbia', 
    'XK': 'Kosovo'
}

# Extract first two letters of bus names and map to country names
yearly_avg_df.columns = [iso2_to_country.get(bus[:2], bus[:2]) for bus in yearly_avg_df.columns]
print(yearly_avg_df)


csv_path = os.path.join(os.path.dirname(cwd), "yearly_solar_capacity_factors.csv")
yearly_avg_df.to_csv(csv_path, index=False)


#%% Close datasets
for d in datasets:
    d.close()

# %%
