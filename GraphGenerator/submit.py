import htcondor
import os
import itertools
import time
import shutil
import glob

max_materialize = 495

def wait_for_complete(wait_time, constraint, schedd, itemdata, submit_result,sub_job):
    time.sleep(1)
    print(constraint)
    while True:
        ads = schedd.query(
            constraint=constraint,
            projection=["ClusterId", "ProcId", "Out", "JobStatus"],
        )
        if len(itemdata) == 0: return
        if len(ads) < max_materialize:
            sub_data = itemdata[:max_materialize - len(ads)]
            print(len(itemdata))
            submit_result += [schedd.submit(sub_job, itemdata=iter(sub_data))]
            print(f"==> Submitting {len(sub_data)} jobs to cluster {submit_result[-1].cluster()}")
            itemdata = itemdata[max_materialize - len(ads):]
            constraint = '||'.join([f'ClusterId == {id.cluster()}' for id in submit_result])
            print(len(itemdata))
            # print(constraint)

        n_runs = len([ad["JobStatus"] for ad in ads if ad["JobStatus"] == 2])
        n_idle = len([ad["JobStatus"] for ad in ads if ad["JobStatus"] == 1])
        n_other = len([ad["JobStatus"] for ad in ads if ad["JobStatus"] > 2])
        print(f"-- {n_idle} idle, {n_runs}/{init_N} running ({len(itemdata)} left)... (wait for another {wait_time} seconds)")
        if n_other > 0:
            print(f"-- {n_other} jobs in other status, please check")
        if n_other > 0 and (n_runs + n_idle == 0):
            print(f"-- {n_other} jobs in other status, other's done, please check")
            return

        time.sleep(wait_time)


cur_dir=os.getcwd()
schedd = htcondor.Schedd()

# submit jobs using htcondor
sub_job = htcondor.Submit({
        "executable": "/lustre/collider/mocen/software/condaenv/hailing/bin/python",
        "arguments": f"./generate_graph_from_ROOT.py -i $(in_file) -o $(out_file) -c $(channel) -sr $(sr)",
        "output": f"log/$(job_tag)_$(cur).out",
        "error": f"log/$(job_tag)_$(cur).err",
        # "log": f"log/$(job_tag)_$(ProcID).log",
        "log": f"log/$(job_tag)_$(ClusterID).log",
        "rank": '(OpSysName == "CentOS")',
        'initialdir': "./",
        "getenv": 'True',
})

data_path_format = '/lustre/collider/mocen/project/bbtautau/hhard/lxplus-sync/sample_{channel}/*.root'
out_path_format = '/lustre/collider/mocen/project/bbtautau/machinelearning/traindir/{channel}/{sr}/split/{sample_name}.pt'
channels = [
    'hadhad',
    # 'lephadSLT'
]
SRs = [
    'ggF_high', 
    'ggF_low', 
    'vbf'
]
submit_result = []
itemdata = []
for channel in channels:
    in_files = glob.glob(data_path_format.format(channel=channel))
    for f in in_files:
        sample_name = os.path.basename(f).split('.')[0]
        itemdata += [
            {'cur': f'{sample_name}', 'job_tag': f'gen_dataset_{channel}_{sr}', 'in_file': f, 
            'out_file': out_path_format.format(channel=channel, sr=sr, sample_name=sample_name), 'channel': channel, 'sr': sr} for sr in SRs
        ]

init_N = len(itemdata)

sub_data = itemdata[:max_materialize]
submit_result += [schedd.submit(sub_job, itemdata=iter(sub_data))]
print(f"==> Submitting {len(sub_data)} jobs to cluster {submit_result[-1].cluster()}")

itemdata = itemdata[max_materialize:]
print(len(itemdata))

constraint = '||'.join([f'ClusterId == {id.cluster()}' for id in submit_result])

# waiting for job complete
# wait_for_complete(15, submit_result.cluster(), schedd, itemdata)
wait_for_complete(30, constraint, schedd, itemdata, submit_result, sub_job)
