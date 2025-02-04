# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
This rule downloads the load data from `Open Power System Data Time series
<https://data.open-power-system-data.org/time_series/>`_. For all countries in
the network, the per country load timeseries are extracted from the dataset.
After filling small gaps linearly and large gaps by copying time-slice of a
given period, the load data is exported to a ``.csv`` file.
"""

import logging

import numpy as np
import pandas as pd
from _helpers import configure_logging, get_snapshots, set_scenario_config
from pandas import Timedelta as Delta

logger = logging.getLogger(__name__)


def load_timeseries(fn, years, countries):
    """
    Read load data from OPSD time-series package version 2020-10-06.

    Parameters
    ----------
    years : None or slice()
        Years for which to read load data (defaults to
        slice("2018","2019"))
    fn : str
        File name or url location (file format .csv)
    countries : listlike
        Countries for which to read load data.

    Returns
    -------
    load : pd.DataFrame
        Load time-series with UTC timestamps x ISO-2 countries
    """
    df = (
        pd.read_csv(fn, index_col=0, parse_dates=[0], date_format="%Y-%m-%dT%H:%M:%SZ")
        .tz_localize(None)
        .dropna(how="all", axis=0)
        .rename(columns={"GB_UKM": "GB"})
        .filter(items=countries)
        .loc[years]
    )
    print(f"[DEBUG] Loaded data shape: {df.shape}")
    print(f"[DEBUG] NaNs per country after load_timeseries:\n{df.isna().sum()}")
    return df


def consecutive_nans(ds):
    cn = (
        ds.isnull()
        .astype(int)
        .groupby(ds.notnull().astype(int).cumsum()[ds.isnull()])
        .transform("sum")
        .fillna(0)
    )
    # Debug: print summary statistics for consecutive NaNs in the series
    max_gap = cn.max()
    print(f"[DEBUG] For series '{ds.name}', max consecutive NaNs: {max_gap}")
    return cn



def fill_large_gaps(ds, shift):
    """
    Fill up large gaps with load data from the previous week.

    This function fills gaps ragning from 3 to 168 hours (one week).
    """
    shift = Delta(shift)
    nhours = shift / np.timedelta64(1, "h")
    if (consecutive_nans(ds) > nhours).any():
        logger.warning(
            "There exist gaps larger then the time shift used for copying time slices."
        )
        # print(
        #     f"[DEBUG] Series '{ds.name}': Found gaps larger than {nhours} hours: {cn[cn > nhours].values}"
        # )
    time_shift = pd.Series(ds.values, ds.index + shift)
    filled = ds.where(ds.notnull(), time_shift.reindex_like(ds))
    n_nans_after = filled.isna().sum()
    print(f"[DEBUG] After fill_large_gaps for series '{ds.name}', remaining NaNs: {n_nans_after}")
    return filled



def nan_statistics(df):
    def max_consecutive_nans(ds):
        return (
            ds.isnull()
            .astype(int)
            .groupby(ds.notnull().astype(int).cumsum())
            .sum()
            .max()
        )

    consecutive = df.apply(max_consecutive_nans)
    total = df.isnull().sum()
    max_total_per_month = df.isnull().resample("m").sum().max()
    stats = pd.concat(
        [total, consecutive, max_total_per_month],
        keys=["total", "consecutive", "max_total_per_month"],
        axis=1,
    )
    print(f"[DEBUG] NaN statistics:\n{stats}")
    return stats


def copy_timeslice(load, cntry, start, stop, delta, fn_load=None):
    start = pd.Timestamp(start)
    stop = pd.Timestamp(stop)
    print(f"[DEBUG] Copying time slice for country '{cntry}' from {start} to {stop} using delta {delta}")
    if start in load.index and stop in load.index:
        if start - delta in load.index and stop - delta in load.index and cntry in load:
            load.loc[start:stop, cntry] = load.loc[
                start - delta : stop - delta, cntry
            ].values
            print(f"[DEBUG] Successfully copied timeslice for {cntry} from {start-delta} to {stop-delta}")
        elif fn_load is not None and cntry in load:
            duration = pd.date_range(freq="h", start=start - delta, end=stop - delta)
            load_raw = load_timeseries(fn_load, duration, [cntry])
            load.loc[start:stop, cntry] = load_raw.loc[
                start - delta : stop - delta, cntry
            ].values
            print(f"[DEBUG] Loaded and copied timeslice from external file for {cntry}")
        else:
            print(f"[DEBUG] Could not find required indices for copying timeslice for {cntry}.")
    else:
        print(f"[DEBUG] Indices {start} or {stop} not in load.index for country '{cntry}'.")



def manual_adjustment(load, fn_load, countries):
    """
    Adjust gaps manual for load data from OPSD time-series package.

     1. For years later than 2015 for which the load data is mainly taken from the
        ENTSOE power statistics

     Kosovo (KV)should be XK and Albania (AL) do not exist in the data set. Kosovo gets the
     same load curve as Serbia and Albania the same as Macdedonia, both scaled
     by the corresponding ratio of total energy consumptions reported by
     IEA Data browser [0] for the year 2013.


     2. For years earlier than 2015 for which the load data is mainly taken from the
        ENTSOE transparency platforms

     Albania (AL) and Macedonia (MK) do not exist in the data set. Both get the
     same load curve as Montenegro,  scaled by the corresponding ratio of total energy
     consumptions reported by  IEA Data browser [0] for the year 2016.

     [0] https://www.iea.org/data-and-statistics?country=WORLD&fuel=Electricity%20and%20heat&indicator=TotElecCons

    Bosnia and Herzegovina (BA) does not exist in the data set for 2019. It gets the
    electricity consumption data from Croatia (HR) for the year 2019, scaled by the
    factors derived from https://energy.at-site.be/eurostat-2021/

    Parameters
    ----------
     load : pd.DataFrame
         Load time-series with UTC timestamps x ISO-2 countries
    load_fn: str
         File name or url location (file format .csv)

    Returns
    -------
     load : pd.DataFrame
         Manual adjusted and interpolated load time-series with UTC
         timestamps x ISO-2 countries
    """
    print("[DEBUG] Starting manual_adjustment")
    if "AL" not in load and "AL" in countries:
        if "ME" in load:
            load["AL"] = load.ME * (5.7 / 2.9)
            print("[DEBUG] Created AL from ME with scaling factor 5.7/2.9")
        elif "MK" in load:
            load["AL"] = load["MK"] * (4.1 / 7.4)
            print("[DEBUG] Created AL from MK with scaling factor 4.1/7.4")

    if "MK" in countries and "MK" in countries:
        if "MK" not in load or load.MK.isnull().sum() > len(load) / 2:
            if "ME" in load:
                load["MK"] = load.ME * (6.7 / 2.9)
                print("[DEBUG] Created or replaced MK from ME with scaling factor 6.7/2.9")

    if "BA" not in load and "BA" in countries:
        if "ME" in load:
            load["BA"] = load.HR * (11.0 / 16.2)
            print("[DEBUG] Created BA from HR with scaling factor 11.0/16.2")


    if "XK" not in load and "XK" in countries:
        if "RS" in load:
            load["XK"] = load["RS"] * (4.8 / 27.0)
            print("[DEBUG] Created XK from RS with scaling factor 4.8/27.0")

    copy_timeslice(load, "GR", "2015-08-11 21:00", "2015-08-15 20:00", Delta(weeks=1))
    copy_timeslice(load, "AT", "2018-12-31 22:00", "2019-01-01 22:00", Delta(days=2))
    copy_timeslice(load, "CH", "2010-01-19 07:00", "2010-01-19 22:00", Delta(days=1))
    copy_timeslice(load, "CH", "2010-03-28 00:00", "2010-03-28 21:00", Delta(days=1))
    # is a WE, so take WE before
    copy_timeslice(load, "CH", "2010-10-08 13:00", "2010-10-10 21:00", Delta(weeks=1))
    copy_timeslice(load, "CH", "2010-11-04 04:00", "2010-11-04 22:00", Delta(days=1))
    copy_timeslice(load, "NO", "2010-12-09 11:00", "2010-12-09 18:00", Delta(days=1))
    # whole january missing
    copy_timeslice(
        load,
        "GB",
        "2010-01-01 00:00",
        "2010-01-31 23:00",
        Delta(days=-365),
        fn_load,
    )
    # 1.1. at midnight gets special treatment
    copy_timeslice(
        load,
        "IE",
        "2016-01-01 00:00",
        "2016-01-01 01:00",
        Delta(days=-366),
        fn_load,
    )
    copy_timeslice(
        load,
        "PT",
        "2016-01-01 00:00",
        "2016-01-01 01:00",
        Delta(days=-366),
        fn_load,
    )
    copy_timeslice(
        load,
        "GB",
        "2016-01-01 00:00",
        "2016-01-01 01:00",
        Delta(days=-366),
        fn_load,
    )

    copy_timeslice(load, "BG", "2018-10-27 21:00", "2018-10-28 22:00", Delta(weeks=1))
    copy_timeslice(load, "LU", "2019-01-02 11:00", "2019-01-05 05:00", Delta(weeks=-1))
    copy_timeslice(load, "LU", "2019-02-05 20:00", "2019-02-06 19:00", Delta(weeks=-1))

    if "UA" in countries:
        copy_timeslice(
            load, "UA", "2013-01-25 14:00", "2013-01-28 21:00", Delta(weeks=1)
        )
        copy_timeslice(
            load, "UA", "2013-10-28 03:00", "2013-10-28 20:00", Delta(weeks=1)
        )

    print("[DEBUG] Finished manual_adjustment")
    return load


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_electricity_demand")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day
    )

    fixed_year = snakemake.params["load"].get("fixed_year", False)
    years = (
        slice(str(fixed_year), str(fixed_year))
        if fixed_year
        else slice(snapshots[0], snapshots[-1])
    )

    interpolate_limit = snakemake.params.load["interpolate_limit"]
    countries = snakemake.params.countries

    time_shift = snakemake.params.load["time_shift_for_large_gaps"]
    print(f"[DEBUG] Input used: {snakemake.input.reported}")


    print(f"[DEBUG] Loading data for years {years} and countries {countries}")
    load = load_timeseries(snakemake.input.reported, years, countries)

    load = load.reindex(index=snapshots)
    print(f"[DEBUG] After reindexing, NaNs per country:\n{load.isna().sum()}")


    if "UA" in countries:
        # attach load of UA (best data only for entsoe transparency)
        load_ua = load_timeseries(snakemake.input.reported, "2018", ["UA"])
        snapshot_year = str(snapshots.year.unique().item())
        time_diff = pd.Timestamp("2018") - pd.Timestamp(snapshot_year)
        # hack indices (currently, UA is manually set to 2018)
        load_ua.index -= time_diff
        load["UA"] = load_ua
        print(f"[DEBUG] Attached UA load; NaNs for UA: {load['UA'].isna().sum()}")
        # attach load of MD (no time-series available, use 2020-totals and distribute according to UA):
        # https://www.iea.org/data-and-statistics/data-browser/?country=MOLDOVA&fuel=Energy%20consumption&indicator=TotElecCons
        if "MD" in countries:
            load["MD"] = 6.2e6 * (load_ua / load_ua.sum())
            print(f"[DEBUG] Created MD load from UA data; NaNs for MD: {load['MD'].isna().sum()}")


    if snakemake.params.load["manual_adjustments"]:
        load = manual_adjustment(load, snakemake.input[0], countries)
        print(f"[DEBUG] After manual_adjustment, NaNs per country:\n{load.isna().sum()}")


    logger.info(f"Linearly interpolate gaps of size {interpolate_limit} and less.")
    load = load.interpolate(method="linear", limit=interpolate_limit)
    print(f"[DEBUG] After linear interpolation, NaNs per country:\n{load.isna().sum()}")

    logger.info(f"Filling larger gaps by copying time-slices of period '{time_shift}'.")
    load = load.apply(fill_large_gaps, shift=time_shift)
    print(f"[DEBUG] After fill_large_gaps, NaNs per country:\n{load.isna().sum()}")

    if snakemake.params.load["supplement_synthetic"]:
        logger.info("Supplement missing data with synthetic data.")
        fn = snakemake.input.synthetic
        synthetic_load = pd.read_csv(fn, index_col=0, parse_dates=True)

        # Check if the column "KV" exists, and rename it to "XK"
        if "KV" in synthetic_load.columns:
            logger.info("Renaming column 'KV' to 'XK' in synthetic load data.")
            synthetic_load = synthetic_load.rename(columns={"KV": "XK"})
        countries_synth = list(set(countries) - set(["UA", "MD"]))
        # # UA, MD, XK do not appear in synthetic load data
        # countries = list(set(countries) - set(["UA", "MD", "XK"]))

        synthetic_load = synthetic_load.loc[snapshots, countries]
        logger.info(f"Synthetic load data summary:\n{synthetic_load.describe()}")
        load = load.combine_first(synthetic_load)
        print(f"[DEBUG] After supplementing synthetic data, NaNs per country:\n{load.isna().sum()}")

#    Final check for missing values
    if load.isna().any().any():
        print(f"[ERROR] Final load data still contains NaNs:\n{load.isna().sum()}")
    assert not load.isna().any().any(), (
        "Load data contains nans. Adjust the parameters "
        "`time_shift_for_large_gaps` or modify the `manual_adjustment` function "
        "for implementing the needed load data modifications."
    )

    # need to reindex load time series to target year
    if fixed_year:
        load.index = load.index.map(lambda t: t.replace(year=snapshots.year[0]))
        print(f"[DEBUG] After reindexing to fixed_year, index sample: {load.index[:5]}")

    load.to_csv(snakemake.output[0])
    print("[DEBUG] Load data successfully written to output.")