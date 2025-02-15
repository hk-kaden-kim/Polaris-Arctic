import netCDF4, os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio

from datetime import datetime

def sec_to_date(sec, format="%A, %B %d, %Y %I:%M:%S"):
    return datetime.fromtimestamp(sec).strftime(format)

def plot_arctic_data(Axes, data, lons, lats, vmin_vmax=None, cmap='jet', area=[-180, 180, 58, 90], zorder=10):

    # Set viz area
    Axes.set_extent(area, ccrs.PlateCarree())

    # Map EO values
    if vmin_vmax:
        im = Axes.pcolormesh(lons, lats, data, zorder=zorder, 
                             vmin=vmin_vmax[0], vmax=vmin_vmax[1],
                             transform=ccrs.PlateCarree(), cmap=cmap, alpha=0.2)
    else:
        im = Axes.pcolormesh(lons, lats, data, zorder=zorder,
                             transform=ccrs.PlateCarree(), cmap=cmap, alpha=0.2)
    
    # Visualize map features
    Axes.gridlines(color='#C6C6C6', zorder=10)

    Axes.add_feature(cfeature.OCEAN, facecolor='#303030', zorder=0)
    Axes.add_feature(cfeature.LAND,  facecolor='#303030', zorder=5)
    Axes.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#C6C6C6', zorder=5)
    Axes.add_feature(cfeature.COASTLINE, edgecolor='#C6C6C6', linewidth=2, zorder=5)

    return im

def load_era5_data(root, MAX_WSPD=40, MAX_SWH=10):

    eo_data = dict()

    ncf_name = os.path.join(root, "data_stream-oper_stepType-instant.nc")
    assert os.path.isfile(ncf_name), f"File not exist : {ncf_name}"

    ncf = netCDF4.Dataset(ncf_name)

    lats = np.array(ncf.variables.get('latitude'))
    lons = np.array(ncf.variables.get('longitude'))

    n_lats, n_lons = len(lats), len(lons)
    lats = np.tile(lats, (n_lons, 1)).T
    lons = np.tile(lons, (n_lats, 1))

    eo_data["lats"] = lats
    eo_data["lons"] = lons

    sic = np.array(ncf.variables.get('siconc'))

    u10 = np.array(ncf.variables.get('u10'))
    v10 = np.array(ncf.variables.get('v10'))
    wspd = np.sqrt(np.add(np.square(u10[:,:,:]),np.square(v10[:,:,:])))
    wspd_masked = np.where(np.isnan(sic), np.nan, wspd)

    eo_data["wind_speed"] = wspd
    eo_data["wind_speed_masked"] = wspd_masked
    eo_data['wind_speed_max'] = MAX_WSPD
    eo_data["sea_ice_conc"] = sic

    ncf_name = os.path.join(root, "data_stream-wave_stepType-instant.nc")
    assert os.path.isfile(ncf_name), f"File not exist : {ncf_name}"

    ncf = netCDF4.Dataset(ncf_name)

    wave = np.array(ncf.variables.get('swh'))
    wave_resize = np.repeat(wave, 2, axis=2).repeat(2, axis=1)[:,:-1,:]
    wave_resize = np.where(sic>0, 0, wave_resize)
    wave_resize_masked = np.where(np.isnan(sic), np.nan, wave_resize)

    eo_data["wave_height"] = wave_resize
    eo_data["wave_height_masked"] = wave_resize_masked
    eo_data['wave_height_max'] = MAX_SWH

    return eo_data

def load_arcnet_data(root, tif_name='arcnet_epsg4326_25e-2.tif'):

    arcnet_data = dict()

    # tif_name = os.path.join(root, 'arcnet_rasterized_warp_epsg4326.tif')
    tif_name = os.path.join(root, tif_name)

    with rasterio.open(tif_name) as src:

        transform = src.transform
        left, bottom, right, top = src.bounds

        lons = np.arange(left, right, transform[0])
        lats = np.arange(bottom, top+transform[0], transform[0])[::-1]
        n_lats, n_lons = len(lats), len(lons)
        lats = np.tile(lats, (n_lons, 1)).T
        lons = np.tile(lons, (n_lats, 1))

        # Read the first band
        data = src.read(1)  

        # ignore non-pac area
        data = np.where(data==0, np.nan, data)

        # match the size of data
        nan_row = np.full((1, data.shape[1]), np.nan)
        data = np.vstack((data, nan_row))

    arcnet_data['lats'] = lats
    arcnet_data['lons'] = lons
    arcnet_data['pac'] = data
    arcnet_data['pac_filter'] = [i+1 for i in range(83)]

    return arcnet_data

def risk_calc(ship_info, era5_data, arcnet_data):

    risk_info = {"date_idx":[d-1001 for d in ship_info['date']],
                "lats":era5_data['lats'], "lons":era5_data['lons'],
                "w_risk":[],
                "wspd":[], "swh":[], "sic":[], "pac":[]}

    for i,d_i in enumerate(risk_info['date_idx']):

        # Normalize
        wspd = era5_data['wind_speed_masked'][d_i,:,:] / era5_data['wind_speed_max']
        swh = era5_data['wave_height_masked'][d_i,:,:] / era5_data['wave_height_max']
        sic = era5_data['sea_ice_conc'][d_i,:,:] 

        if i == 0:
            pac_raw = arcnet_data['pac']
            pac_raw = np.where(np.isnan(pac_raw) & ~np.isnan(sic), 0, pac_raw) # Set 0 to Ocean
            pac = np.where(~np.isnan(pac_raw) & ~np.isin(pac_raw, ship_info['pac_mask']), 0, 1) # Set 0 to non-selected pacs

        # Weighted sum
        weights = [0.25, 0.25, 0.25, 0.25]
        risk = wspd * weights[0] + swh * weights[1] + sic * weights[2] + pac * weights[3]

        # Update entire risk values
        risk_info['w_risk'].append(risk)
        risk_info['wspd'].append(wspd)
        risk_info['swh'].append(swh)
        risk_info['sic'].append(sic)
        risk_info['pac'].append(pac)

        # Find the index of the nearest value
        lons = ship_info['lons'][i] if ship_info['lons'][i] < 180 else ship_info['lons'][i] - 360
        lats = ship_info['lats'][i]
        lons_idx = np.abs(risk_info["lons"][0,:] - lons).argmin()
        lats_idx = np.abs(risk_info["lats"][:,0] - lats).argmin()

        # Update Shipping route risk value
        ship_info['w_risk'].append(risk_info["w_risk"][i][lats_idx, lons_idx])
        ship_info['wspd'].append(risk_info["wspd"][i][lats_idx, lons_idx])
        ship_info['swh'].append(risk_info["swh"][i][lats_idx, lons_idx])
        ship_info['sic'].append(risk_info["sic"][i][lats_idx, lons_idx])
        ship_info['pac'].append(risk_info["pac"][i][lats_idx, lons_idx])

    return risk_info, ship_info