import xarray as xr
import pandas as pd
import datetime as dt
import time 
from netCDF4 import Dataset, date2num, num2date
from bokeh.plotting import figure, output_file, save
from bokeh.io import gridplot, vplot, reset_output


def bokeh_plot(site, obs, obs_label, data_dict, outfile, title, resample_frequency, add_statistics=True):
    '''
    Plots a bokeh (interactive) daily or weekly series. 
    Inputs:
    site: a string ID which is input to the obs and data_dict objects
    obs: a pandas dataframe with columns labeled with the site IDs
    data_dict: an ordered dictionary with keys each of the series you want to plot with structure:
    {'series_name': {'data': pandas_dataframe, 'color': 'the color you want the timeseries to plot as'}}
    outfile: path where you want to save the bokeh plot
    title: title string for your plot
    resample_frequency: either 'w' for weekly or 'd' for daily (this is used to decide the frequency of the plotting)
    add_statistics: if True (default) will include the statistics of KGE and bias in the plot's legend
    '''
    reset_output()
    resample_dict = {'W': 'Weekly',
                'D': 'Daily'}
    output_file(outfile, title=title)

    p1 = figure(width=1000, height=400, x_axis_type = "datetime", title=title)
    obs_resampled =obs.resample(resample_frequency, how='mean')
    p1.line(obs_resampled.index.values,obs_resampled.values/100000, color='#000000',  legend=obs_label, line_width=2)
    
    for scenario in data_dict.keys():
        sim = data_dict[scenario]['data'][site]
        daily_kge = str(np.round(kge(sim, obs, 'D'), 2))
        weekly_kge = str(np.round(kge(sim, obs, 'W'), 2))
        scenario_bias = str(np.round(bias(sim, obs),2))
        if add_statistics:
            legend_label = scenario+', Daily KGE='+daily_kge+',\nWeekly KGE='+weekly_kge+' Bias='+scenario_bias+'%'
        else:
            legend_label = scenario
        sim_resampled = data_dict[scenario]['data'][site].resample(resample_frequency, how='mean')
        p1.line(sim_resampled.index.values, sim_resampled.values/100000, 
            color=data_dict[scenario]['color'], line_width=2, 
                legend=legend_label)
    p1.grid.grid_line_alpha=0.3
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Streamflow [CFS*100,000]'
    p1.legend.label_text_font_size = '8'

    window_size = 30
    window = np.ones(window_size)/float(window_size)
    g = vplot(p1)
    save(g)    



def get_extent(data, limits):
    '''
    This function will make your colorbar dynamically have 
    extend arrows on the end of it, depending on whether or not
    your data exceeds the limits prescribed by your vmax and 
    vmin values.
    '''
    if min(data) < min(limits) and max(data) > max(limits):
        return 'both'
    elif min(data) < min(limits):
        return 'min'
    elif max(data) > max(limits):
        return 'max'
    else:
        return 'neither'

def datestamp():
    
    '''
    This function returns a datestamp of form YYMMDD,
    No inputs required- just call datestamp()
    '''
    
    return dt.datetime.fromtimestamp(time.time()).strftime('%Y%m%d')

def convert_monthly_to_wy(monthly_groupby_df):
    
    '''
    Inputs: a monthly group-by dataframe with 12 values, following the Jan-Dec monthly indices.
    Outputs: a monthly group-by dataframe with 12 values, following the Oct-Sep monthly indices.
    '''
    
    import pandas as pd
    import numpy as np
    
    label=monthly_groupby_df
    oct_dec = monthly_groupby_df[9:12]
    jan_sep = monthly_groupby_df[0:9]
    df = pd.concat([oct_dec, jan_sep])
    df2 = pd.DataFrame(df)
    df2['months'] = pd.Series([4,5,6,7,8,9,10,11,12,1,2,3],  index=monthly_groupby_df.index)
    df2 = df2.set_index('months')
    series = pd.Series(np.squeeze(df2.values), index = range(1,13))
    
    return series

def routed_nc_to_df(ncfile, date_start, date_end, timename='time'):
    ''' read RVIC routed streamflow netcdf file

        Input arguments: the routed netcdf file
        Return: dataframe with your netcdf info
    '''

    d = {}
    nc = Dataset(ncfile, 'r')
#     time =  num2date(nc.variables[timename][0:-1],
#                      nc.variables[timename].units,
#                      nc.variables[timename].calendar)
    
    
    index = pd.date_range(date_start, date_end)

    data = nc.variables['streamflow'][:]
    
    for i, name in enumerate(nc.variables['outlet_name'][:]):
        compressed_name = ''.join(name.compressed())
        d[compressed_name] = pd.Series(data[:-1, i], index=index)
    df = pd.DataFrame(d)
    conversion_cms_cfs = 1000. * 1000. * 1000./(12. * 12. * 12. * 25.4 * 25.4 * 25.4)

    df *= conversion_cms_cfs
    
    return df

def read_nrni_nc(ncfile, timename='Time'):
    
    '''
    
    Input: routed NRNI file in CFS
    
    Output: dataframe containing timeseries of CFS streamflow at NRNI points
    
    '''
    
    from netCDF4 import Dataset, date2num, num2date
    import pandas as pd

    timename = 'Time'
    nc = Dataset(ncfile, 'r')
    time = num2date(nc.variables[timename][:],
                         nc.variables[timename].units,
                         nc.variables[timename].calendar)
    site_indices = nc.variables['Index'][:]
    sites = nc.variables['IndexNames'][:]
    d={}
    for i in site_indices:
        d[sites[i]]=pd.Series(nc.variables['Streamflow'][:,i], index=time)
    df = pd.DataFrame(d)
    
    return df
def load_routed_streamflow(ncfile):
    '''
    This function reads in an RVIC output streamflow timeseries file
    and loads it into a dataframe.
    
    Input:
    RVIC netcdf file
    
    Output:
    Pandas dataframe with routed streamflow locations. 
    '''
    # Read in the netcdf file
    ds = xr.open_dataset(ncfile)
    
    # Make the stream outlet IDs the index names as opposed to integers
    ds.outlets.values = [outlet.decode() for outlet in ds.outlet_name.values]
    
    # Convert into a pandas dataframe
    df = ds.streamflow.to_dataframe()
    
    # Drop a junk timestep (can remove this step once RVIC PR#86 is approved)
    streamflow = df.unstack().streamflow
    
    return streamflow

def read_nrni_nc(ncfile, timename='Time'):
    timename = 'Time'
    nc = Dataset(ncfile, 'r')
    time = num2date(nc.variables[timename][:],
                         nc.variables[timename].units,
                         nc.variables[timename].calendar)
    site_indices = nc.variables['Index'][:]
    sites = nc.variables['IndexNames'][:]
    d={}
    for i in site_indices:
        d[sites[i]]=pd.Series(nc.variables['Streamflow'][:,i], index=time)
    df = pd.DataFrame(d)
    
    return df

def convert_cms_to_cfs(cms):
    cfs = cms*(100*100*100)/(2.54*2.54*2.54*12*12*12)
    return cfs



def water_year(date):
    year = date.year
    if date.month > 9:
        year += 1
    return year
