#!/usr/bin/python3
# run_task_upon_load 升级版，采用线程池自动控制负载

import os
import argparse
import subprocess
import multiprocessing
import psutil
import mongodb_tools
import utils

from tqdm.autonotebook import tqdm
from task_server import TaskManager


CWD = os.getcwd()

def cp_model(model, rip):
    res = os.system(f"scp {CWD}/bin/{model} {rip}:/home/zeal4u/extraspace/Project/NewChampSim/bin/")
    # if res != 0:
        # exit(-1)

def save_to_db(result_dir, model, workloads):
    re = os.system(f"xz -zef {result_dir}/*-{model}.txt")
    if re == 0:
        mongodb_tools.insert_or_overwrite_results(mongodb_tools.get_collection(), [model], workloads, result_dir)
        # os.system(f"scp ./{result_dir}/*{model}* IntelBase:/home/zeal4u/extraspace/Project/champ-sim-explorer/{result_dir}/")


def build_command(model, workload, n_warm, n_sim, is_cloud):
    if is_cloud:
        TRACE_DIR = f"{CWD}/cloudsuite"
    else:
        TRACE_DIR = f"{CWD}/total_traces"
    
    multi = False
    if isinstance(workload, list):
        multi = True
        if len(set(workload)) == 1:
            target_file = f"results_{len(workload)}core_{n_sim}M/mix0-{workload[0]}-{model}.txt" 
        else:
            target_file = f"results_{len(workload)}core_{n_sim}M/mix-{hash(''.join(map(lambda x: x.replace('.champsimtrace.xz', ''), workload)))}-{model}.txt" 

        traces = ' '.join([f"{TRACE_DIR}/{t}" for t in workload])
        result_dir = f"results_{len(workload)}core_{n_sim}M"
        run_model_command = f"({CWD}/bin/{model} -warmup_instructions {n_warm}000000 -simulation_instructions \
            {n_sim}000000 {'-c' if is_cloud else ''} -traces {traces}) &> {target_file}"
    else:
        result_dir = f"results_{n_sim}M"
        target_file = f"results_{n_sim}M/{workload}-{model}.txt"
        run_model_command = f"({CWD}/bin/{model} -warmup_instructions {n_warm}000000 -simulation_instructions\
             {n_sim}000000 {'-c' if is_cloud else ''} -traces {TRACE_DIR}/{workload}) &> {target_file}"
        save_command = f"{CWD}/mongodb_tools.py {model} {workload}"
    
    # os.makedirs(result_dir, exist_ok=True)
    xz_command = f"xz -zef {target_file}"
    workload = workload if isinstance(workload, str) else " ".join(workload)
    save_command = f"{CWD}/mongodb_tools.py {model} {workload} {'-m' if multi else ''} -r {result_dir}"
    # backup_command = f"scp {target_file+'.xz'} IntelBase:~/extraspace/Project/MyChampSim/{result_dir}/"
    return f"{run_model_command} && {xz_command} && {save_command}", target_file


def yield_command(model, workloads, n_warm, n_sim, cover, is_cloud):
    collection = mongodb_tools.get_collection()
    for i, workload in enumerate(workloads):
        command, target_file = build_command(model, workload, n_warm, n_sim, is_cloud)
        trace_key = target_file[target_file.find('/')+1:target_file.find('-bimodal')]
        # print(trace_key)
        if cover or not collection.find_one({'model': utils.clear_str(model), 'data.trace': utils.clear_str(trace_key)}):
            # print(command)
            # res = pool.apply_async(subprocess.run, args=(command,), kwds={"shell":True})
            yield command
    # if is_multi:
    # return results, result_dir

def run(model, workloads, n_warm, n_sim, cover, is_cloud):
    parent = psutil.Process()
    parent.nice(-10)
    with multiprocessing.Pool() as pool:
        tasks = []
        for command in yield_command(model, workloads, n_warm, n_sim, cover, is_cloud):
            tasks.append(pool.apply_async(subprocess.run, args=(command,), kwds={"shell":True}))
        for task in tqdm(tasks):
            task.wait()

def run_on_server(model, workloads, n_warm, n_sim, cover, is_cloud, ip_address=''):
    TaskManager.register("get_queue")
    manager = TaskManager(address=(ip_address, 50000), authkey=b'jsz1995')
    manager.connect()
    task_queue = manager.get_queue()

    if ip_address != '':
        cp_model(model, ip_address)

    # send work start
    for command in yield_command(model, workloads, n_warm, n_sim, cover, is_cloud):
        print(command)
        task_queue.put(f"{command}")
    # task_queue.put(f"{os.getpid()}")
    # send work finish
    # over_msg = task_queue.get()

def litmus_run(model):
    command, target_file = build_command(model, "srv_408.champsimtrace.xz", 1, 1, False)
    command = command.replace(f"&> {target_file}", "")
    os.system(command[: command.find("&&")])

def shell_call():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="The complete model name.")
    parser.add_argument("-w", "--workloads", type=str, default="", 
                        help='The workloads that the models experimented.')
    parser.add_argument("-nw", "--n_warm", type=int, default=1, help="The instruction number of warming up.")
    parser.add_argument("-ns", "--n_sim", type=int, default=1, help="The instruction number of warming up.")
    parser.add_argument("-c", "--cover", action='store_const', const=True, default=False, help="Overwrite the history.")
    parser.add_argument("--cloud", action='store_const', const=True, default=False, help="It is running on cloudsuites.")
    parser.add_argument("-p", "--prior", action='store_const', const=True, default=False, help="Run in high priority.")
    parser.add_argument("-r", "--rip", action='store_const', const="192.168.3.13", default="", help="The remote machine IP.")
    parser.add_argument("-t", "--test", action='store_const', const=True, default=False, help="A litmus test for a model.")

    args = parser.parse_args()
    # print(args)
    if args.test:
        litmus_run(args.model)
    elif args.prior:
        workloads = getattr(utils, args.workloads)
        run(args.model, workloads, args.n_warm, args.n_sim, args.cover, args.cloud)
    else:
        workloads = getattr(utils, args.workloads)
        run_on_server(args.model, workloads, args.n_warm, args.n_sim, args.cover, args.cloud, args.rip)

if __name__ == "__main__":
    shell_call()
