import numpy as np
import pandas as pd
import os
import copy

# For plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import AutoMinorLocator
 
font = {'family': 'serif',
        'weight': 'normal', 'size': 12}
plt.rc('font', **font)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

default_color_list = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

class PlotContainer:
    def __init__(self, scale_vars=1., logx=False, logy=False, xlabel='', xlim=None, ylabel='', ylim=None,
            title='', ax=None, fig=None, figname=None, **kwargs) -> None:
        self.__dict__ = dict(kwargs)
        for name, value in vars().items():
            if (name != 'self') and (name != 'kwargs'):
                setattr(self,name,value)
        if ax==None:
            self.fig, self.ax = plt.subplots(figsize=(7, 5), dpi=600, constrained_layout=True)

    def apply_settings(self, if_legend=False):
        if self.logy:
            self.ax.set_yscale('log')
        else:
            # Adding minor ticks on the y-axis without labels
            self.ax.yaxis.set_minor_locator(AutoMinorLocator())
        if self.logx:
            self.ax.set_xscale('log')
        if self.xlim!=None:
            self.ax.set_xlim(self.xlim)
        if self.ylim!=None:
            self.ax.set_ylim(self.ylim)
        if if_legend:
            self.ax.legend()

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    def savefig(self, figname=None):
        if figname!=None:
            self.fig.savefig(figname)
        else:
            self.fig.savefig(self.figname)

class RatioPlotContainer(PlotContainer):
    def __init__(self, scale_vars=1, logx=False, logy=False, xlabel='', xlim=None, ylabel='', ylim=None, title='', axes=None, fig=None, figname=None, **kwargs) -> None:
        if axes==None:
            fig, axes = plt.subplots(2, 1, figsize=(6, 5), dpi=600, constrained_layout=True, sharex='col', gridspec_kw={'height_ratios': [3, 1]})
        super().__init__(scale_vars, logx, logy, xlabel, xlim, ylabel, ylim, title, axes[0], fig, figname, **kwargs)
        self.ax_main = axes[0]
        self.ax_ratio = axes[1]
        self.y_max, self.y_min = 1.1, 0.9

        # ditc to save the x and y values e.g. {'wp77': [ [xvalues], [[yvalues0], [yvalues1], [color]] ]}
        self.dict_label_xyvalues = dict()

    def insert_data_with_array(self, ary, bin_edges, ary_index:int, label:str, weights:list=None, color:str=None):
        # insert a data list
        # ary_index = 0 or 1
        if self.dict_label_xyvalues.get(label) is None:
            # self.dict_label_xyvalues[label] = [None, [None, None, None]]
            self.dict_label_xyvalues[label] = dict(
                x_values=None, color=None,
                y_values0=None, y_values1=None, 
                y_err0=None, y_err1=None,
            )
        y_value, bin_edges = np.histogram(ary, bins=bin_edges, weights=weights)
        y_err = None 
        if type(weights) != type(None):
            y_err, bin_edges = np.histogram(ary, bins=bin_edges, weights=np.array(weights)**2)
            y_err = y_err**0.5
        else:
            y_err = np.sqrt(y_value)
        
        # Save value
        self.dict_label_xyvalues[label]['color'] = color
        self.dict_label_xyvalues[label]['x_values'] = (bin_edges[:-1]+bin_edges[1:])/2,
        self.dict_label_xyvalues[label][f'y_values{ary_index}'] = y_value
        self.dict_label_xyvalues[label][f'y_err{ary_index}'] = y_err

    def insert_data(self, x_values, y_value, ary_index:int, label:str, add_if_exist:bool=True, y_err=None, color:str=None):
        # insert a data from existing values
        # ary_index = 0 or 1
        if (self.dict_label_xyvalues.get(label) is None):
            # Insert a new item
            self.dict_label_xyvalues[label] = dict(
                x_values=None, color=None,
                y_values0=None, y_values1=None, 
                y_err0=None, y_err1=None,
            )
            # Save value
            if color is not None:
                self.dict_label_xyvalues[label]['color'] = copy.copy(color)
            self.dict_label_xyvalues[label]['x_values'] = copy.copy(x_values)
            self.dict_label_xyvalues[label][f'y_values{ary_index}'] = copy.copy(y_value)
            self.dict_label_xyvalues[label][f'y_err{ary_index}'] = copy.copy(y_err)
        elif (self.dict_label_xyvalues.get(label).get(f'y_values{ary_index}') is None) or (not add_if_exist):
            # Insert / replace an item
            if color is not None:
                self.dict_label_xyvalues[label]['color'] = copy.copy(color)
            self.dict_label_xyvalues[label][f'y_values{ary_index}'] = copy.copy(y_value)
            self.dict_label_xyvalues[label][f'y_err{ary_index}'] = copy.copy(y_err)
        else:
            # add y_value
            if color is not None:
                self.dict_label_xyvalues[label]['color'] = copy.copy(color)
            self.dict_label_xyvalues[label]['x_values'] = copy.copy(x_values)
            self.dict_label_xyvalues[label][f'y_values{ary_index}'] += y_value
            y_value = self.dict_label_xyvalues[label][f'y_values{ary_index}'] 
            # merge y_error
            tmp_err = self.dict_label_xyvalues[label][f'y_err{ary_index}']
            if type(y_err)!=type(None) or type(tmp_err)!=type(None):
                y_err = np.sqrt(y_err**2 + tmp_err**2)
                self.dict_label_xyvalues[label][f'y_err{ary_index}'] = y_err


    def draw_ratio(self, ratio_ylabel='Ratio', draw_error=True, fmt='none', print_ratio=False, **kwargs):
        # Draw ratio plot
        for key, val in self.dict_label_xyvalues.items():
            # Obtain data
            x_values = val['x_values']
            y_values0, y_values1 = val['y_values0'], val['y_values1']
            y_err0, y_err1 = val['y_err0'], val['y_err1']
            color = val['color']

            # Calculate ratio
            ratio = y_values1.copy()
            mask = (y_values0!=0) & (y_values1!=0)
            if mask.sum()==0:
                continue
            ratio[mask] /= y_values0[mask]
            ratio[~mask] = np.nan
            if print_ratio:
                print(f"Ratio of {key}: {ratio}")

            self.ax_ratio.step(x_values, ratio, where='mid', color=color, label=key, **kwargs)
            if draw_error:
                y_rel_err = np.zeros_like(ratio)
                y_rel_err[mask] = np.sqrt( (y_err0[mask]/y_values0[mask])**2 + (y_err1[mask]/y_values1[mask])**2 ) 
                yerr = y_rel_err * ratio
                self.ax_ratio.errorbar(x_values, ratio, yerr=yerr, fmt=fmt, color=color, **kwargs)
            self.y_max = max(self.y_max, max(ratio[mask] + (yerr[mask] if draw_error else 0)) * 1.1)
            self.y_min = min(self.y_min, min(ratio[mask] - (yerr[mask] if draw_error else 0)) * 0.9)

        self.ax_ratio.axhline(y=1, linestyle='--', color='black', linewidth=0.5)
        self.ax_ratio.set_ylabel(ratio_ylabel)


    def apply_settings(self, if_legend=True, ratio_ylim=None):
        super().apply_settings(if_legend=True)
        self.ax.set_xlabel('')
        self.ax_ratio.set_xlabel(self.xlabel)
        self.ax_ratio.yaxis.set_minor_locator(AutoMinorLocator())
        if self.xlim!=None:
            self.ax_ratio.set_xlim(self.xlim)

        if ratio_ylim==None:
            # valid_ratio = self.ratio[~np.isnan(self.ratio)]
            self.ax_ratio.set_ylim(self.y_min, self.y_max)
        else:
            self.ax_ratio.set_ylim(ratio_ylim)

        if if_legend:
            self.ax_ratio.legend(fontsize=7)



"""
Draw Images
"""
def drawCurve(x, y, linelable=None, logx=False, logy=False, xlabel='', ylabel='', title='', ax=None, fig=None):
    if ax==None:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=400, constrained_layout=True)

    ax.plot(x, y, label=linelable)
    if linelable!=None:
        ax.legend()
    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')
        
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plt.tight_layout() 
    return (ax, fig)

def drawArrayAs1DHist(arr, nbins=20, logx=False, logy=False, xlabel='', title='', filename=None, draw_quantile=True, density=False):
    arr = np.array(arr)
    print(f'mean: {np.mean(abs(arr))}. median: {np.median(abs(arr))}. std: {np.std(abs(arr))}. max: {abs(arr).max()}')
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 4), dpi=400)
    ax.axvline(x=np.median(arr), linestyle='--', label=f'median={np.median(arr):.2f}', color="xkcd:red")
    if draw_quantile:
        ax.axvline(x=np.quantile(arr, 0.68), linestyle='--', label='68%', color="xkcd:sky blue")
        ax.axvline(x=np.quantile(arr, 0.95), linestyle='--', label='95%', color="xkcd:salmon")
    plt.hist(arr, nbins, density=density)
    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    ax.legend(title=f'data size: {len(arr)}')
    plt.tight_layout()
    if filename!=None:
        plt.savefig(filename)
    else:
        plt.show()

def drawArraysAs2DHist(arr1, arr2, bins=30, logx=False, logy=False, logz=False, xlabel='', ylabel='', title='', filename=None):
    hist, xedges, yedges = np.histogram2d(arr1, arr2, bins=bins)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=500)
    X, Y = np.meshgrid(xedges, yedges)
    norm = colors.LogNorm() if logz else None
    cax = ax.pcolormesh(X, Y, hist.T, norm=norm)
    fig.colorbar(cax, ax=ax, label='Counts in bin')
    ax.set_xlabel(xlabel)
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if filename!=None:
        fig.savefig(filename)
    else:
        fig.show()

# by Fan Hu
class Resolution:
    def __init__(self, x_bin: np.ndarray, df_xy: pd.DataFrame) -> None:
        """
        x_bin: bin edge for x-axis. generated with np.linspace(3, 6, 31) np.logspace(3,6,50)
        df_xy: reconstruction result. with columns: ["x_value", "y_value"]
        """
        self.x_bin = x_bin
        self.bin_center = (self.x_bin[1:] + self.x_bin[:-1]) / 2
        self.n_bins = len(self.x_bin) - 1

        self.df_xy = df_xy
        self.num_samples = len(df_xy)
        self.get_resolution()

    def get_resolution(self):
        self.df_resolution = pd.DataFrame(columns=["tot_5", "tot_50", "tot_95", "tot_16", "tot_84", "mean", "std"],
            index=np.arange(self.n_bins), dtype="float")
        x_value = self.df_xy.x_value.to_numpy()
        for i in range(self.n_bins):
            bin_low, bin_up = self.x_bin[i], self.x_bin[i+1]
            xval_mask = (x_value >= bin_low) & (x_value < bin_up)
            
            events_ = self.df_xy.loc[xval_mask]
            if(len(events_)==0):
                self.df_resolution.iloc[i] = [np.nan] * len(self.df_resolution.columns)
                continue
            q5, q50, q95 = np.quantile(events_["y_value"], [0.05, 0.5, 0.95])
            q16, q84 = np.quantile(events_["y_value"], [0.16, 0.84])
            std = np.std(events_["y_value"])
            mean = np.mean(events_["y_value"])
            self.df_resolution.iloc[i] = q5, q50, q95, q16, q84, mean, std
    
    def plot(self, xlabel=r'Neutrino energy [GeV]', median_label=r"Median Prediction Error",
            ylabel=r'Resolution', logx=True, title=None, ax=None, fig=None):
        if ax==None:
            fig, ax = plt.subplots(figsize=(5, 4), dpi=400, constrained_layout=True)
        ax.plot(self.bin_center, self.df_resolution["tot_50"], 
            color="xkcd:red", label=median_label)
        ax.fill_between(self.bin_center, 
            self.df_resolution["tot_5"], self.df_resolution["tot_95"], 
            # alpha=0.5, color="xkcd:sky blue")
            color="#FFFF00")
        ax.fill_between(self.bin_center, 
            self.df_resolution["tot_16"], self.df_resolution["tot_84"], 
            # alpha=0.5, color="xkcd:sky blue")
            color="#00FF00")
        ax.legend(title=f"data size: {self.num_samples}")
        
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale('log')
        ax.set_xlim(self.x_bin[0], self.x_bin[-1])
        ax.set_ylabel(ylabel)
        if title != None:
            ax.set_title(title)
        # plt.tight_layout() 
        return (ax, fig)


class Resolution2D:
    def __init__(self, x_bin: np.ndarray, y_bin: np.ndarray, df_xyz: pd.DataFrame, quantile=0.5) -> None:
        """
        x_bin: bin edge for x-axis. 
        y_bin: bin edge for y-axis. 
        df_xyz: reconstruction result. with columns: ["x_value", "y_value", "z_value"]
        """
        self.x_bin = x_bin
        self.y_bin = y_bin
        self.x_bin_center = (self.x_bin[1:] + self.x_bin[:-1]) / 2
        self.y_bin_center = (self.y_bin[1:] + self.y_bin[:-1]) / 2

        self.df_xyz = df_xyz
        self.get_resolution(quantile)
        
    def get_resolution(self, quantile):
        self.resolution = []
        self.mesh_x, self.mesh_y = np.meshgrid(self.x_bin_center, self.y_bin_center)
        self.median_z = np.zeros_like(self.mesh_x)
        self.plus_sigma, self.minus_sigma = np.zeros_like(self.mesh_x), np.zeros_like(self.mesh_x)
        # Calculate the median pred_error for each cell in the grid
        for i, x_val in enumerate(self.x_bin_center):
            for j, y_val in enumerate(self.y_bin_center):
                # Filter pred_error values for the current pair (r_val, energy_val)
                mask = (self.df_xyz.y_value<self.y_bin[j+1]) & (self.df_xyz.y_value>=self.y_bin[j]) & \
                    (self.df_xyz.x_value<self.x_bin[i+1]) & (self.df_xyz.x_value>=self.x_bin[i])
                z = self.df_xyz.z_value[mask]
                if z.size > 0:
                    self.median_z[j, i] = np.quantile(z, quantile)
                    self.plus_sigma[j, i] = np.quantile(z, 0.84)
                    self.minus_sigma[j, i] = np.quantile(z, 0.16)
                else:
                    self.median_z[j, i] = np.nan
                    self.plus_sigma[j, i] = np.nan
                    self.minus_sigma[j, i] = np.nan

                # self.median_z[j, i] = np.quantile(z, 0.8) if z.size > 0 else np.nan
    
    def plot(self, xlabel=r'Neutrino energy [GeV]', ylabel=r'Vertex Distance [m]', median_label=r"Median Prediction Error", 
        log_x=True, log_y=False, title=None, ax=None, fig=None, zmin=None, zmax=None):
        if ax==None:
            fig, ax = plt.subplots(figsize=(5, 4), dpi=400, constrained_layout=True)
        
        z = self.median_z[~np.isnan(self.median_z)]
        zmax = z.max() if zmax==None else zmax
        zmin = z.min() if zmin==None else zmin
        c = plt.pcolormesh(self.x_bin_center, self.y_bin_center, self.median_z, shading='auto', 
            norm=colors.LogNorm(vmin=zmin, vmax=zmax))
        fig.colorbar(c, label=median_label)
        ax.set_xlabel(xlabel)
        if log_x:
            ax.set_xscale('log')
        ax.set_xlim(self.x_bin[0], self.x_bin[-1])
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale('log')
        if title != None:
            ax.set_title(title)
        # plt.tight_layout() 
        return (ax, fig)

    def plot_unc(self, xlabel=r'Neutrino energy [GeV]', ylabel=r'Vertex Distance [m]', z_label=r"Relative Uncertainty", 
        log_x=True, log_y=False, title=None, ax=None, fig=None, zmin=None, zmax=None):
        if ax==None:
            fig, ax = plt.subplots(figsize=(5, 4), dpi=400, constrained_layout=True)

        z = (self.plus_sigma - self.minus_sigma) / self.median_z
        zmax = 1 if zmax==None else zmax
        zmin = 0 if zmin==None else zmin
        c = plt.pcolormesh(self.x_bin_center, self.y_bin_center, z, shading='auto', 
            norm=colors.Normalize(vmin=zmin, vmax=zmax))
        fig.colorbar(c, label=z_label)
        ax.set_xlabel(xlabel)
        if log_x:
            ax.set_xscale('log')
        ax.set_xlim(self.x_bin[0], self.x_bin[-1])
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale('log')
        if title != None:
            ax.set_title(title)
        return (ax, fig)

        