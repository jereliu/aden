import tqdm

from netCDF4 import Dataset
import pandas as pd
import numpy as np

# Randall data: netCDF format


def netcdf2csv(file_year=2015, file_addr='./data/real/randall/',
               latbounds=[36, 48], lonbounds=[-83, -67]):
    read_name = file_addr + '/nc/%d.nc' % file_year
    writ_name = file_addr + '/csv/%d.csv' % file_year

    Randall_raw = Dataset(read_name)

    # degrees east ?
    lats = Randall_raw.variables['LAT'][:]
    lons = Randall_raw.variables['LON'][:]
    pm25 = Randall_raw.variables['PM25'][:]

    # latitude lower and upper index
    latui = np.argmin(np.abs(lats - latbounds[0]))
    latli = np.argmin(np.abs(lats - latbounds[1]))

    # longitude lower and upper index
    lonli = np.argmin(np.abs(lons - lonbounds[0]))
    lonui = np.argmin(np.abs(lons - lonbounds[1]))

    # convert to long format
    Randall_list = []

    print("\nProcessing year %d..\n" % file_year)

    for lat_id in tqdm.tqdm(range(latli, latui)):
        for lon_id in range(lonli, lonui):
            pm25_val = pm25[lat_id, lon_id]
            if not np.isnan(pm25_val):
                lat_val = lats[lat_id]
                lon_val = lons[lon_id]
                Randall_list.append([lat_val, lon_val, pm25_val])

    Randall_final = \
        pd.DataFrame(np.array(Randall_list), columns=["lat", "lon", "pm25"])
    Randall_final.to_csv(writ_name, index=False)


