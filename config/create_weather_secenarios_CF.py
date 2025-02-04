# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import itertools
import os 

print('hello')

if "snakemake" in globals():
    print('hello1')
    filename = snakemake.output[0]
else:
    print('Hello3')
    filename = "weather_scenarios_cutouts.yaml"

template = """
weather_year_{year}:
  snapshots:
    start: "{year}-01-01 00:00"
    end: "{year}-12-31 23:00"
    inclusive: "both"
  atlite:
    default_cutout: europe-{year}-era5
    cutouts:
      europe-{year}-era5:
        module: era5
        x: [-12., 42.]
        y: [33., 72]
        dx: 0.3
        dy: 0.3
        time: ['{year}', '{year}']
  renewable:
    onwind:
      cutout: europe-{year}-era5
    offwind-ac:
      cutout: europe-{year}-era5
    offwind-dc:
      cutout: europe-{year}-era5
    offwind-float:
      cutout: europe-{year}-era5
    solar:
      cutout: europe-{year}-era5
    solar-hsat:
      cutout: europe-{year}-era5
    hydro:
      cutout: europe-{year}-era5
  solar_thermal:
    cutout: europe-{year}-era5
  sector:
    heat_demand_cutout: europe-{year}-era5
  lines:
    dynamic_line_rating:
      cutout: europe-{year}-era5
"""

first_year = 2015
last_year = 2021

# Generate config values for individual years
config_values = dict(year=range(first_year, last_year + 1))

combinations = [
    dict(zip(config_values.keys(), values))
    for values in itertools.product(*config_values.values())
]

print('It still works')
print(f"Writing to: {filename}")
print("Current working directory:", os.getcwd())

# Write individual cutouts for each year
with open(filename, "w") as f:
    for config in combinations:
        f.write(template.format(**config))
        print(f"Writing weather scenario for {config['year']}")
