import function_library as fl
from raincoat.disdrometer import read_parsivel as parsivel
import xarray as xr
import scipy
import numpy as np

# for pop up figures:
import matplotlib
matplotlib.use('TkAgg')

parsivel_file = '../../data/20190222_disdrometer.nc'
radar_file = '../../data/20190222-punta-arenas-limrad94.nc'


# read in the disdrometer data using the function from raincoat
d_data = parsivel.readPars(parsivel_file, var_Ze="radar_reflectivity", var_time="time", var_vmean="fall_velocity",
                       var_dmean="number_concentration", var_rr="rainfall_rate", transpose=True)

# read in the radar data using the xarray library
r_data = xr.open_mfdataset(radar_file, combine='by_coords')
range_index = fl.argnearest(r_data["range"].values, 250)
print(f'index of range gate closest to 250 m: {range_index} ({r_data["range"].values[range_index]})')

times_Wband = r_data.time  # xarray DataArray, time in seconds since 2001-01-01
times_parsivel = d_data[0].unixtime  # pandas timeseries, unixtime is seconds since 1970-01-01 00:00:00

# convert the time stamp of the radar to the same time format as the disdrometer
times_Wband = fl.seconds_since_2001_to_unix(times_Wband.values)

# extract the rain rates in m/s and convet to mm/h
rainrates = d_data[0].rainrate * 3600/1e-3

# interpolate Disdrometer timeseries to W Band radar timestamp:
interp_rr = scipy.interpolate.interp1d(times_parsivel, rainrates, fill_value="extrapolate")
rr_new = interp_rr(times_Wband)

Z_250 = r_data.Ze.values[range_index, :]


# make dictionaries for the plotting function
RR = {}
RR['var'] = rr_new
RR['system'] = 'Parsivel'
RR['var_unit'] = 'mm/h'
RR['name'] = 'rain rate'
RR['mask'] = np.isnan(Z_250)

Ze_W = {}
Ze_W['var'] = fl.lin2z(Z_250)
Ze_W['mask'] = np.isnan(Z_250)
Ze_W['var_unit'] = 'dBZ'
Ze_W['name'] = 'reflectivity'
Ze_W['system'] = "LIMRAD94"

fig, ax = fl.plot_scatter(RR, Ze_W, identity_line=False)
ax.set_ylim([-12, 31])
ax.axhline(y=19, linewidth=4, color='salmon')
fig.show()
fig.savefig('plots/Parsivel_scatter.png')

Ze_W['mask'] = np.logical_or(Ze_W['mask'], (Ze_W['var'] < 10))
Ze_W['mask'] = np.logical_or(Ze_W['mask'],  (RR['var'] < 0.5))
median_Z = np.median(np.ma.masked_array(data=Ze_W['var'], mask=Ze_W['mask']))
RR['mask'] = np.logical_or(RR['mask'], (RR['var'] < 0.5))

fig, ax = fl.plot_scatter(RR, Ze_W, identity_line=False)
ax.set_ylim([10, 20])
ax.axhline(y=19, linewidth=4, color='salmon')
ax.text(0.73, 0.93, 'median = {:5.3f}'.format(median_Z),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.axhline(y=median_Z, linewidth=1, color='black')
fig.show()
fig.savefig('plots/Parsivel_scatter2')
