import numpy as np
import pandas as pd
import torch
import os
import torch_geometric

# For plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
 
font = {'family': 'serif',
        'weight': 'normal', 'size': 12}
plt.rc('font', **font)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def get_batch_dataList(dataIdList, batchDir):
    batch = []
    for idx in dataIdList:
        data = torch.load(os.path.join(batchDir, f'batch_{idx}.pt'))
        batch += data.to_data_list()
    return torch_geometric.data.Batch.from_data_list(batch)

def getAngleTheta(v1, v2):
    v1 = v1.reshape(-1,3)
    v1 = v1 / np.linalg.norm(v1, axis=1).reshape(-1,1)
    v2 = v2.reshape(-1,3)
    v2 = v2 / np.linalg.norm(v2, axis=1).reshape(-1,1)
    diff = ((v1 - v2) ** 2).sum(axis=1).clip(0, 4)
    return np.arccos( 1 - 1./2 * (diff) ) / np.pi * 180


def getAngularError(truth_with_pred):
    truth_with_pred = truth_with_pred.reshape(-1,6)
    truth = truth_with_pred[:,:3]
    pred = truth_with_pred[:,3:]
    return getAngleTheta(truth, pred)


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
        plt.yscale('log')
    if logx:
        plt.xscale('log')
        
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plt.tight_layout() 
    return (ax, fig)

def drawArrayAs1DHist(arr, nbins=20, logx=False, logy=False, xlabel='', title='', filename=None, draw_quantile=True, density=False):
    arr = np.array(arr)
    print(f'mean: {np.mean(abs(arr))}. median: {np.median(abs(arr))}. std: {np.std(abs(arr))}. max: {abs(arr).max()}')
    plt.clf()
    fig = plt.subplots(figsize=(5, 4), dpi=400)
    plt.axvline(x=np.median(arr), linestyle='--', label=f'median={np.median(arr):.2f}', color="xkcd:red")
    if draw_quantile:
        plt.axvline(x=np.quantile(arr, 0.68), linestyle='--', label='68%', color="xkcd:sky blue")
        plt.axvline(x=np.quantile(arr, 0.95), linestyle='--', label='95%', color="xkcd:salmon")
    plt.hist(arr, nbins, density=density)
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    plt.title(title)
    plt.xlabel(xlabel)

    plt.legend(title=f'data size: {len(arr)}')
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
        plt.savefig(filename)
    else:
        plt.show()

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
        self.df_resolution = pd.DataFrame(columns=["tot_2.4", "tot_50", "tot_97.6", "tot_16", "tot_84", "std", "counts"],
            index=np.arange(self.n_bins), dtype="float")
        x_value = self.df_xy.x_value.to_numpy()
        for i in range(self.n_bins):
            bin_low, bin_up = self.x_bin[i], self.x_bin[i+1]
            xval_mask = (x_value >= bin_low) & (x_value < bin_up)
            
            events_ = self.df_xy.loc[xval_mask]
            if(len(events_)==0):
                self.df_resolution.iloc[i] = [np.nan] * 7
                continue
            q2, q50, q97 = np.quantile(events_["y_value"], [0.024, 0.5, 0.976])
            q16, q84 = np.quantile(events_["y_value"], [0.16, 0.84])
            std = np.std(events_["y_value"])
            counts = len(events_)
            self.df_resolution.iloc[i] = q2, q50, q97, q16, q84, std, counts
    
    def plot(self, xlabel=r'Neutrino energy [GeV]', median_label=r"Median Prediction Error",
            ylabel=r'Resolution', logx=True, title=None, ax=None, fig=None):
        if ax==None:
            fig, ax = plt.subplots(figsize=(5, 4), dpi=400, constrained_layout=True)
        ax.plot(self.bin_center, self.df_resolution["tot_50"], 
            color="xkcd:red", label=median_label)
        ax.fill_between(self.bin_center, 
            self.df_resolution["tot_2.4"], self.df_resolution["tot_97.6"], 
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
        self.quantile_97_6, self.quantile_2_4 = np.zeros_like(self.mesh_x), np.zeros_like(self.mesh_x)
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
                    self.quantile_97_6[j, i] = np.quantile(z, 0.976)
                    self.quantile_2_4[j, i] = np.quantile(z, 0.024)
                else:
                    self.median_z[j, i] = np.nan
                    self.plus_sigma[j, i] = np.nan
                    self.minus_sigma[j, i] = np.nan
                    self.quantile_97_6[j, i] = np.nan
                    self.quantile_2_4[j, i] = np.nan

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

        