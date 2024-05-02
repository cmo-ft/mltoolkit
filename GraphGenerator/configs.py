import math
import torch
from collections import namedtuple

data_path_format = '/lustre/collider/mocen/project/bbtautau/hhard/lxplus-sync/sample_{channel}/*.root'
out_path_format = '/lustre/collider/mocen/project/bbtautau/machinelearning/traindir/{channel}/{sr}/split/{sample_name}.pt'
# not use data/hhttbbL10/hhttbbVBFNONSM
ignore_files = ['data.root', 'hhttbbL10.root', 'hhttbbVBFNONSM.root']

ParticleProperty = namedtuple("ParticleProperty", ["Importance", "Loc"])
GraphSetting = namedtuple(
    'GraphSetting',
    ['channel', 'SR', 'tree_name', 'particles', 'edges', 'node_feature', 'edge_feature', 'glob_feature']
)

common_particles = {
    "bbtt_HH": ParticleProperty(Importance=1, Loc=[0, 0]),
    "bbtt_HH_vis": ParticleProperty(Importance=0.5, Loc=[0, 0]),
    "bbtt_H_bb": ParticleProperty(Importance=1, Loc=[math.sqrt(3)/2, 0.5]),
    "bbtt_mmc": ParticleProperty(Importance=1, Loc=[-math.sqrt(3)/2, 0.5]),
    "met_NOSYS": ParticleProperty(Importance=1, Loc=[0, -1]),
    "bbtt_H_vis_tautau": ParticleProperty(Importance=0.5, Loc=[-math.sqrt(3)/2, 0.5]),
    "bbtt_mmc_nu1": ParticleProperty(Importance=1, Loc=[-math.sqrt(3)/2, -0.5]),
    "bbtt_mmc_nu2": ParticleProperty(Importance=1, Loc=[-math.sqrt(3)/2, -0.5]),
    "bbtt_Tau1": ParticleProperty(Importance=1, Loc=[-1, math.sqrt(3)]),
    "bbtt_Jet_b1": ParticleProperty(Importance=1, Loc=[1, math.sqrt(3)]), # importance is decided by pcbt_GN2v01
    "bbtt_Jet_b2": ParticleProperty(Importance=1, Loc=[1, math.sqrt(3)]), # importance is decided by pcbt_GN2v01
}
common_node_feature_format = ["{p}_eta", "{p}_phi", "{p}_pt_NOSYS", "{p}_E", "{p}_m"]
common_edge_feature_format = ["dR_{p0}{p1}", "dPhi_{p0}{p1}", "M_{p0}{p1}",]


def get_node_edge_features(node_feature_format_list, edge_feature_format_list, particles):
    node_features = []
    edges = [[], []]
    edge_features = []
    for i, particle_i in enumerate(particles):
        # construct edge and edge feature
        for j, particle_j in enumerate(particles):
            if i==j:
                continue
            edges[0].append(j)
            edges[1].append(i)
            edge_features.extend([edge_ft_format.format(p0=particle_i, p1=particle_j) for edge_ft_format in edge_feature_format_list])
        node_features.extend([ft_format.format(p=particle_i) for ft_format in node_feature_format_list])
    return node_features, torch.tensor(edges, dtype=torch.long), edge_features


def get_graph_setting(channel, SR):
    config = None
    # graph setting of hadhad channel
    if channel=='hadhad':
        hadhad_pars = common_particles.copy()
        hadhad_pars['bbtt_Tau2'] = ParticleProperty(Importance=1, Loc=[-1, math.sqrt(3)])
        node_features, edges, edge_features = get_node_edge_features(common_node_feature_format,common_edge_feature_format, hadhad_pars.keys())
        if SR=='ggF_high':
            glob_features = [
                "num_jets", "T1", "mTtau1", "spher_bbtt", 
                # "pflow_bbtt",  # currently value of this var is all 0. Remove it.
                "cent_bbtt", "met_NOSYS_sumet"
            ]
            tree_name = 'tree_2tag_OS_LL_GGFSR_350mHH'
        elif SR=='ggF_low':
            glob_features = [
                "num_jets", "mEff_ttj", "mHHStar", "T2", "spher_bbtt", "cent_bbtt", "HT", "MT2"
            ]
            tree_name = 'tree_2tag_OS_LL_GGFSR_0_350mHH'
        elif SR=='vbf':
            glob_features = [
                "num_jets", "num_jets", "vbf_eta0eta1", "thrust_ttjf"
            ]
            tree_name = 'tree_2tag_OS_LL_VBFSR'
            # vbf SR contains 2 vbf jets
            for i in range(2):
                hadhad_pars[f'vbfjet{i}'] = ParticleProperty(Importance=1, Loc=[0, 1])
            node_features, edges, edge_features = get_node_edge_features(common_node_feature_format,common_edge_feature_format, hadhad_pars.keys())

        config = GraphSetting(
            channel=channel, SR=SR, tree_name=tree_name, 
            particles=hadhad_pars, edges=edges, node_feature=node_features, edge_feature=edge_features,
            glob_feature=glob_features
        )

    # # graph setting of lephad SLT channel
    # elif channel=='lephadSLT':
    #     lephadSLT_pars = common_particles.copy()
    #     lephadSLT_pars['bbtt_Lepton1'] = ParticleProperty(Importance=1, Loc=[-1, math.sqrt(3)])
    #     node_features, edges, edge_features = get_node_edge_features(common_node_feature_format,common_edge_feature_format, lephadSLT_pars.keys())
    #     if SR=='ggF_high':
    #         glob_features = [
    #             "num_jets", "mTW", "HT", "T1", "mTtau", "met_NOSYS_sumet"
    #         ]
    #         tree_name = 'tree_2tag_OS_TL_GGFSR_350mHH'
    #     elif SR=='ggF_low':
    #         glob_features = [
    #             "num_jets", "mTW", "T1"
    #         ]
    #         tree_name = 'tree_2tag_OS_TL_GGFSR_0_350mHH'
    #     config = GraphSetting(
    #         channel=channel, SR=SR, tree_name=tree_name, 
    #         particles=hadhad_pars, edges=edges, node_feature=node_features, edge_feature=edge_features,
    #         glob_feature=glob_features
    #     )

    return config