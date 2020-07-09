# File with a collection of functions from larda

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import datetime
from copy import copy

def test_function(file):
    print(file)

def argnearest(array, value):
    """find the index of the nearest value in a sorted array
    for example time or range axis

    Args:
        array (np.array): sorted array with values, list will be converted to 1D array
        value: value to find
    Returns:
        index
    """
    if type(array) == list:
        array = np.array(array)
    i = np.searchsorted(array, value) - 1

    if not i == array.shape[0] - 1:
            if np.abs(array[i] - value) > np.abs(array[i + 1] - value):
                i = i + 1
    return i

def ident(x):
    return x

def raw2Z(array, **kwargs):
    """raw signal units (MRR-Pro) to reflectivity Z"""
    return array * kwargs['wl']**4 / (np.pi**5) / 0.93 * 10**6

def divide_by(val):
    return lambda var: var / val

def lin2z(array):
    """linear values to dB (for np.array or single number)"""
    return 10 * np.ma.log10(array)

def z2lin(array):
    """dB to linear values (for np.array or single number)"""
    return 10 ** (array / 10.)

def plot_scatter(data_container1, data_container2, identity_line=True, **kwargs):
    """scatter plot for variable comparison between two devices or variables

    Args:
        data_container1 (dict): container 1st device
        data_container2 (dict): container 2nd device
        x_lim (list): limits of var used for x axis
        y_lim (list): limits of var used for y axis
        c_lim (list): limits of var used for color axis
        **identity_line (bool): plot 1:1 line if True
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **var_converter (string): alternate name for the z_converter
        **custom_offset_lines (float): plot 4 extra lines for given distance
        **info (bool): print slope, interception point and R^2 value
        **fig_size (list): size of the figure in inches
        **fontsize (int): default: 15
        **fonteight (int): default: semibold
        **colorbar (bool): if True, add a colorbar to the scatterplot
        **color_by (dict): data container 3rd device
        **scale (string): 'lin' or 'log' --> if you get a ValueError from matplotlib.colors
                          try setting scale to lin, log does not work for negative values!
        **cmap (string) : colormap
        **nbins (int) : number of bins for histograms

    Returns:
        ``fig, ax``
    """
    var1_tmp = data_container1
    var2_tmp = data_container2

    combined_mask = np.logical_or(var1_tmp['mask'], var2_tmp['mask'])
    colormap = kwargs['cmap'] if 'cmap' in kwargs else 'viridis'
    if 'var_converter' in kwargs:
        kwargs['z_converter'] = kwargs['var_converter']
    # convert var from linear unit with any converter given in helpers
    if 'z_converter' in kwargs and kwargs['z_converter'] != 'log':
        var1 = get_converter_array(kwargs['z_converter'])[0](var1_tmp['var'][~combined_mask].ravel())
        var2 = get_converter_array(kwargs['z_converter'])[0](var2_tmp['var'][~combined_mask].ravel())
    else:
        var1 = var1_tmp['var'][~combined_mask].ravel()  # +4.5
        var2 = var2_tmp['var'][~combined_mask].ravel()

    x_lim = kwargs['x_lim'] if 'x_lim' in kwargs else [np.nanmin(var1), np.nanmax(var1)]
    y_lim = kwargs['y_lim'] if 'y_lim' in kwargs else [np.nanmin(var2), np.nanmax(var2)]
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [6, 6]
    fig_size[0] = fig_size[0]+2 if 'colorbar' in kwargs and kwargs['colorbar'] else fig_size[0]
    fontweight =  kwargs['fontweight'] if 'fontweight' in kwargs else'semibold'
    fontsize = kwargs['fontsize'] if 'fontsize' in kwargs else 15
    nbins = 120 if not 'nbins' in kwargs else kwargs['nbins']

    # create histogram plot
    s, i, r, p, std_err = stats.linregress(var1, var2)
    H, xedges, yedges = np.histogram2d(var1, var2, bins=nbins, range=[x_lim, y_lim])

    if 'color_by' in kwargs:
        print("Coloring scatter plot by {}...\n".format(kwargs['color_by']['name']))
        # overwrite H
        H = np.zeros(H.shape)
        var3 = kwargs['color_by']['var'][~combined_mask].ravel()
        # get the bins of the 2d histogram using digitize
        x_coords = np.digitize(var1, xedges)
        y_coords = np.digitize(var2, yedges)
        # find unique bin combinations = pixels in scatter plot

        # sort x and y coordinates using lexsort
        # lexsort sorts by multiple columns, first by y_coords then by x_coords

        newer_order = np.lexsort((x_coords, y_coords))
        x_coords = x_coords[newer_order]
        y_coords = y_coords[newer_order]
        var3 = var3[newer_order]
        first_hit_y = np.searchsorted(y_coords, np.arange(1, nbins+2))
        first_hit_y.sort()
        first_hit_x = [np.searchsorted(x_coords[first_hit_y[j]:first_hit_y[j + 1]], np.arange(1, nbins + 2))
                    + first_hit_y[j] for j in np.arange(nbins)]

        for x in range(nbins):
            for y in range(nbins):
                H[y, x] = np.nanmedian(var3[first_hit_x[x][y]: first_hit_x[x][y + 1]])

    X, Y = np.meshgrid(xedges, yedges)
    fig, ax = plt.subplots(1, figsize=fig_size)

    if not 'scale' in kwargs or kwargs['scale']=='log':
       formstring = "%.2E"
       if not 'c_lim' in kwargs:
            pcol = ax.pcolormesh(X, Y, np.transpose(H), norm=matplotlib.colors.LogNorm(), cmap=colormap)
       else:
            pcol = ax.pcolormesh(X, Y, np.transpose(H), norm=matplotlib.colors.LogNorm(vmin=kwargs['c_lim'][0],
                                                                                      vmax=kwargs['c_lim'][1]),
                                 cmap=colormap)
    elif kwargs['scale'] == 'lin':
        formstring = "%.2f"
        if not 'c_lim' in kwargs:
            kwargs['c_lim'] = [np.nanmin(H), np.nanmax(H)]
        pcol = ax.pcolormesh(X, Y, np.transpose(H), vmin=kwargs['c_lim'][0], vmax=kwargs['c_lim'][1], cmap=colormap)

    if 'info' in kwargs and kwargs['info']:
        ax.text(0.01, 0.93, 'slope = {:5.3f}\nintercept = {:5.3f}\nR^2 = {:5.3f}'.format(s, i, r ** 2),
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontweight=fontweight, labelsize=fontsize)

    # helper lines (1:1), ...
    if identity_line: add_identity(ax, color='salmon', ls='-')

    if 'custom_offset_lines' in kwargs:
        offset = np.array([kwargs['custom_offset_lines'], kwargs['custom_offset_lines']])
        for i in [-2, -1, 1, 2]: ax.plot(x_lim, x_lim + i * offset, color='salmon', linewidth=0.7, linestyle='--')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if 'z_converter' in kwargs and kwargs['z_converter'] == 'log':
        #ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('{} {} [{}]'.format(var1_tmp['system'], var1_tmp['name'], var1_tmp['var_unit']), fontweight=fontweight, fontsize=fontsize)
    ax.set_ylabel('{} {} [{}]'.format(var2_tmp['system'], var2_tmp['name'], var2_tmp['var_unit']), fontweight=fontweight, fontsize=fontsize)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    if 'colorbar' in kwargs and kwargs['colorbar']:
        c_lim = kwargs['c_lim'] if 'c_lim' in kwargs else [1, round(H.max(), int(np.log10(max(np.nanmax(H), 10.))))]
        cmap = copy(plt.get_cmap(colormap))
        cmap.set_under('white', 1.0)
        cbar = fig.colorbar(pcol, use_gridspec=True, extend='min', extendrect=True,
                            extendfrac=0.01, shrink=0.8, format=formstring)
        if not 'color_by' in kwargs:
            cbar.set_label(label="frequency of occurrence", fontweight=fontweight, fontsize=fontsize)
        else:
            cbar.set_label(label="median {} [{}]".format(kwargs['color_by']['name'], kwargs['color_by']['var_unit']), fontweight=fontweight, fontsize=fontsize)
        cbar.set_clim(c_lim)
        cbar.aspect = 50

    if 'title' in kwargs:
        if kwargs['title'] == True:
            ax.set_title(data_container1['paraminfo']['location'] +
                         ts_to_dt(data_container1['ts'][0]).strftime(" %Y-%m-%d %H:%M - ") +
                         ts_to_dt(data_container1['ts'][-1]).strftime("%Y-%m-%d %H:%M"), fontweight=fontweight, fontsize=fontsize)
        else:
            ax.set_title(kwargs['title'], fontweight=fontweight, fontsize=fontsize)

    plt.grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    #ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    if 'colorbar' in kwargs and kwargs['colorbar']:
        cbar.ax.tick_params(axis='both', which='major', labelsize=fontsize-2,
                            width=2, length=4)

    return fig, ax

def add_identity(axes, *line_args, **line_kwargs):
    """helper function for the scatter plot"""
    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def dt_to_ts(dt):
    """datetime to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

def ts_to_dt(ts):
    """unix timestamp to dt"""
    return datetime.datetime.utcfromtimestamp(ts)

def seconds_since_2001_to_unix(times):
    return np.array([x + dt_to_ts(datetime.datetime(2001, 1, 1)) for x in list(times)])

def get_converter_array(string, **kwargs):
    """colletion of converters that works on arrays
    combines time, range and varconverters (i see no conceptual separation here)

    the maskconverter becomes relevant, if the order is no
    time, range, whatever (as in mira spec)

    Returns:
        (varconverter, maskconverter) which both are functions
    """
    if string == 'since20010101':
        return lambda x: x + dt_to_ts(datetime.datetime(2001, 1, 1)), ident
    elif string == 'unix':
        return lambda x: x, ident
    elif string == 'since19691231':
        return lambda x: x + dt_to_ts(datetime.datetime(1969, 12, 31, 23)), ident
    elif string == 'since19700101':
        return lambda x: x + dt_to_ts(datetime.datetime(1970, 1, 1)), ident
    elif string == 'beginofday':
        if 'ncD' in kwargs.keys():
            return (lambda h: (h.astype(np.float64) * 3600. + \
                               float(dt_to_ts(datetime.datetime(kwargs['ncD'].year,
                                                                kwargs['ncD'].month,
                                                                kwargs['ncD'].day)))),
                    ident)
    elif string == "hours_since_year0":
        return (lambda x: x*24*60*60 - 62167305599.99999,
                ident)
    elif string == "pollytime":
        return (lambda x: np.array([x[i,1] + dt_to_ts(datetime.datetime.strptime(str(int(x[i,0])), "%Y%m%d"))\
                for i in range(x.shape[0])]),
                ident)


    elif string == "km2m":
        return lambda x: x * 1000., ident
    elif string == "sealevel2range":
        return lambda x: x - kwargs['altitude'], ident

    elif string == 'z2lin':
        return z2lin, ident
    elif string == 'lin2z':
        return lin2z, ident
    elif string == 'switchsign':
        return lambda x: -x, ident

    elif string == "mira_azi_offset":
        return lambda x: (x + kwargs['mira_azi_zero']) % 360, ident

    elif string == 'transposedim':
        return np.transpose, np.transpose
    elif string == 'transposedim+invert3rd':
        return transpose_and_invert, transpose_and_invert
    elif string == 'divideby2':
        return divide_by(2.), ident
    elif string == 'keepNyquist':
        return ident, ident
    elif string == 'raw2Z':
        return raw2Z(**kwargs), ident
    elif string == "extract_level0":
        return lambda x: x[:, 0], ident
    elif string == "extract_level1":
        return lambda x: x[:, 1], ident
    elif string == "extract_level2":
        return lambda x: x[:, 2], ident
    elif string == 'extract_1st':
        return lambda x: np.array(x[0])[np.newaxis,], ident
    elif string == "none":
        return ident, ident
    else:
        raise ValueError("converter {} not defined".format(string))

def transpose_and_invert(var):
    return np.transpose(var)[:, :, ::-1]
