import htcondor
import os
import itertools
import time
import shutil

max_materialize = 295

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


schedd = htcondor.Schedd()

# submit jobs using htcondor
sub_job = htcondor.Submit({
        "executable": "/lustre/collider/mocen/software/condaenv/hailing/bin/python",
        "arguments": f"./generate_pt.py -i $(inpath)  -o $(outpath)",
        "output": f"log/$(job_tag).out",
        "error": f"log/$(job_tag).err",
        "log": f"log/$(job_tag).log",
        "rank": '(OpSysName == "CentOS")',
        "getenv": 'True',
})



submit_result = []
itemdata = []

from collections import namedtuple
DataConfig = namedtuple("DataConfig", ["path", "train_ids", "val_ids", "test_ids"])
samples = [
    DataConfig(path="/lustre/collider/mocen/project/hailing/data/muons/0.1-1TeV/", train_ids=list(range(0,4)), val_ids=list(range(4,5)), test_ids=list(range(5,7))),
    DataConfig(path="/lustre/collider/mocen/project/hailing/data/muons/1-10TeV/", train_ids=list(range(0,40)), val_ids=list(range(40,45)), test_ids=list(range(45,50))),
    DataConfig(path="/lustre/collider/mocen/project/hailing/data/muons/10-100TeV/", train_ids=list(range(0,80)), val_ids=list(range(80,90)), test_ids=list(range(90,100))),
]
import pathlib
for sample in samples:
    sample_tag = pathlib.Path(sample.path).name
    for i in sample.train_ids:
        itemdata += [
            {'inpath': f'{sample.path}/batch{i}/data/data.root', 'outpath': f'train/{sample_tag}_{i}.pt', 'job_tag': sample_tag + f'_train_{i}'}
        ]
    for i in sample.val_ids:
        itemdata += [
            {'inpath': f'{sample.path}/batch{i}/data/data.root', 'outpath': f'validation/{sample_tag}_{i}.pt', 'job_tag': sample_tag + f'_val_{i}'}
        ]
    for i in sample.test_ids:
        itemdata += [
            {'inpath': f'{sample.path}/batch{i}/data/data.root', 'outpath': f'test/{sample_tag}_{i}.pt', 'job_tag': sample_tag + f'_test_{i}'}
        ]

init_N = len(itemdata)

sub_data = itemdata[:max_materialize]
submit_result += [schedd.submit(sub_job, itemdata=iter(sub_data))]
print(f"==> Submitting {len(sub_data)} jobs to cluster {submit_result[-1].cluster()}")

itemdata = itemdata[max_materialize:]
print(len(itemdata))

constraint = '||'.join([f'ClusterId == {id.cluster()}' for id in submit_result])

# # waiting for job complete
wait_for_complete(30, constraint, schedd, itemdata, submit_result, sub_job)
