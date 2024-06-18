
import glob
import torch_geometric
import pandas as pd
import torch
import time
import json
import numpy as np
import trident
from trident import Hits
import pandas as pd
import yaml
import numpy as np
from torch_geometric.data import HeteroData
import uproot as ur
import torch_geometric.utils as tgu
from torch_geometric.data import Data
import torch_geometric.utils as tgu
import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description="generate .pt file from data.root")
# e.g. /lustre/neutrino/zhangfuyudi/hailing/cascade/sy_data/bl_100tev_gv/10k_p55/job_0/data/data.root
parser.add_argument('-i', '--input', type=str, help="input file name.")
parser.add_argument('-o', '--output', type=str, help="output file name.")
args = parser.parse_args()

t_start = time.time()

RADIUS_DOM = 0.215 #[m]
lepton_pdg = 13
time_limit = 10000
nhits_cut = 15
ndoms_cut = 1

inpath = args.input
outpath = args.output
tts = 3
tts_sigma = tts / (2*np.sqrt(2*np.log(2)))
jitter_sigma = 0.1 / (2*np.sqrt(2*np.log(2)))
c = 0.299792458 #[m/ns]
print(f"From {inpath} to {outpath}")

def event_filter(hits):
    # filter nhits
    flag = len(hits) > nhits_cut
    # Get DOMs & filter ndoms
    doms = hits.groupby('DomId').median()
    doms['nhits'] = hits.t0.groupby('DomId').count()
    doms = doms.sort_values('t0', ascending=True)
    flag = flag and (len(doms) > ndoms_cut)
    return flag, doms

def get_single_graph(file_name: str, entry: int, neu: pd.DataFrame):
    # Construct graph
    graph = Data()

    # get truth info
    primary = ur.open(file_name)['Primary'].arrays(library='pd', entry_start=entry, entry_stop=entry+1)
    if len(primary)==0:
        return None
    primary = primary.sort_values('e0', ascending=False)
    lep = primary.loc[primary.PdgId==lepton_pdg]
    if len(lep)==0:
        return None
    else:
        lep = lep.iloc[0]
    
    lep_direction = torch.tensor(lep[['px', 'py', 'pz']]).type(torch.float).view(1,-1)
    lep_energy = torch.tensor(lep[['e0']]).type(torch.float).view(-1)
    lep_vertex = torch.tensor(lep[['x0','y0','z0']]).type(torch.float).view(1,-1)

    # load pmt hits
    try:
        pmthits = Hits.read_data(file_name, "PmtHit", entry=entry)
        pmthits = Hits.apply_qe(pmthits, "PmtHit").spread(tts_sigma).data
        hits = pmthits
    except Exception as ex:
        print(f"Error: load PMT hits error in file {file_name} entry {entry}: %s. Please check"%ex, flush=True)
        return None
    try:
        # load sipm hits
        sipmhits = Hits.read_data(file_name, "SipmHit", entry=entry)
        sipmhits = Hits.apply_qe(sipmhits, "SipmHit").spread(jitter_sigma).data
        # rearange pmtid and sipm id. 
        # WARNING: 31 is the pmt number. It may be changed
        sipmhits['PmtId'] = sipmhits['SipmId'] + 31
        sipmhits = sipmhits.drop('SipmId', axis=1)

        # merge pmt hits and sipm hits
        hits = pd.concat([pmthits, sipmhits])
    except Exception as ex:
        print(f"Error: load SiPM hits error in file {file_name} entry {entry}: %s. Please check"%ex, flush=True)
        
    hits = hits.loc[hits.t0>0]
    if len(hits)<2:
        return None

    hits = hits.sort_values('t0', ascending=True)

    first_hit = hits.iloc[0]
    tmin = first_hit['t0']
    # graph.tmin = tmin
    hits = hits.loc[hits.t0<(time_limit+tmin)]

    flag, doms = event_filter(hits=hits)
    if not flag:
        return None
    

    doms[['x0', 'y0', 'z0', 't0']] = doms[['x0', 'y0', 'z0', 't0']] - first_hit[['x0','y0','z0', 't0']]
    neu_direction, neu_energy, neu_vertex = None, None, None
    if type(neu)!=type(None):
        neu_momentum = torch.tensor(neu[['px', 'py', 'pz']]).type(torch.float).view(1,-1)
        neu_energy = neu_momentum.norm().view(-1)
        neu_direction = neu_momentum / neu_energy
        neu_vertex = torch.tensor(neu[['x','y','z']]).type(torch.float).view(1,-1)
    

    hits_tr = torch.tensor(doms[['t0', 'x0', 'y0', 'z0']].to_numpy()).type(torch.float32)
    hits_tr[:, 0] *= c
    nhits = torch.tensor(doms[['nhits']].to_numpy()).type(torch.float32).view(-1)
    first_hit = torch.tensor(first_hit[['x0','y0','z0', 't0']].to_numpy()).type(torch.float32).view(-1)

    graph = Data(hits_tr=hits_tr, nhits=nhits, first_hit=first_hit, lep_energy=lep_energy, lep_direction=lep_direction, lep_vertex=lep_vertex, neu_direction=neu_direction, neu_energy=neu_energy, neu_vertex=neu_vertex, file_path=file_name, entry=entry)
    
    return graph


def process_gnn_data():
    """
    Get graph from existed root files
    """

    all_graph = []
    entries = ur.open(inpath)['Primary'].num_entries
    neutrino = []

    mc_events = glob.glob(os.path.dirname(inpath) + '/../mc_events*.json')
    if len(mc_events)>0:
        with open(mc_events[0]) as f:
            js = json.load(f)
        neutrino = pd.json_normalize(js, record_path='particles_in', meta=['event_id']).set_index('event_id')

    for ientry in range(entries):
        graph = get_single_graph(file_name=inpath, entry=ientry, neu=neutrino.iloc[ientry] if len(neutrino)!=0 else None)
        all_graph += [graph] if graph!=None else []

    torch.save(torch_geometric.data.Batch.from_data_list(all_graph), outpath)


process_gnn_data()
print(time.time()-t_start)
