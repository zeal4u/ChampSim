#!/bin/python3

from pymongo import collection
from collections import Counter

import utils
import pymongo
import argparse


__all__ = [
    'get_collection',
    'insert_or_overwrite_results'
]


def get_collection():
    myclient = pymongo.MongoClient("mongodb://192.168.3.13:27017/")
    champsimdb = myclient["new_champsimdb"]
    perfCollection = champsimdb["results"]
    return perfCollection


def insert_or_overwrite_results(collection, curModels: list, workloads: list, targetDir="./results_200M", isOverwrite=False):
    for model in curModels:
        try:
            raw_data = utils.analyze_output(
                targetDir, model, workloads, utils.get_all_results)
            data = utils.transform_to_number(raw_data)
            tmp = {}
            for j in data:
                tmp[utils.clear_str(j)] = data[j]
            if not collection.find_one({'model': utils.clear_str(model)}):
                insert(collection, utils.clear_str(model), tmp)
            elif isOverwrite:
                overwrite(collection, utils.clear_str(model), tmp)
            else:
                update(collection, utils.clear_str(model), tmp)
        except ValueError as e:
            print(model, raw_data)
            raise e


def insert(collection, model, data):
    return collection.insert_one({'model': model, 'data': [{'trace': j, **data[j]} for j in data]})


def overwrite(collection, model, data):
    return collection.update_one({'model': model}, {"$set": {'data': [{'trace': j, **data[j]} for j in data]}})


def update(collection, model, data):
    for j in data:
        if not collection.find_one({'model': model, 'data.trace': j}):
            collection.update_one(
                {'model': model}, {'$push': {'data': {'trace': j, **data[j]}}})
        else:
            collection.update_one({'model': model}, {'$set': {'data.$[idx]': {
                                  'trace': j, **data[j]}}}, upsert=True, array_filters=[{"idx.trace": j}])


def convert2dict(collected_data, workloads):
    data = {}
    for model in collected_data:
        trace_dict = {}
        for trace_data in collected_data[model]['data']:
            flag = False
            # compatible with old records
            flag |= isinstance(workloads[0], str) and trace_data['trace'] in map(lambda x: utils.clear_str(x), workloads)
            flag |= 'Trace Name' in trace_data and trace_data['Trace Name'] in workloads
            if flag: 
                trace_dict[trace_data['trace']] = trace_data
                del trace_dict[trace_data['trace']]['trace']
        data[model] = trace_dict
    return data


def convert2normalized_ipc(ipcs, workloads):
    normalized_ipcs = {}
    reformat = {}
    for model in ipcs:
        tmp = {}
        if ipcs[model]:
            for k in ipcs[model]['data']:
                try:
                    if k['trace'] in workloads:
                        tmp[k['trace']] = k['cumulative IPC']
                except KeyError as e:
                    print(model, k['trace'], "has no cumulative IPC!")
                    raise e
            if len(tmp) != len(workloads):
                for k in workloads:
                    if k not in tmp:
                        print("Can't find results for", k)
                raise Exception(f"Not Enough Results! {model}")
            reformat[model] = tmp
    for model in reformat:
        if model != "bimodal-no-no-no-no-lru-1core":
            n_ipc = {}
            for trace in reformat[model]:
                n_ipc[trace] = reformat[model][trace] / \
                    reformat['bimodal-no-no-no-no-lru-1core'][trace]
            normalized_ipcs[model] = n_ipc
        else:
            # special for baseline
            normalized_ipcs[model] = reformat[model]
    return normalized_ipcs


def shell_call():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="The complete model name.")
    parser.add_argument("workloads", type=str, metavar='N', nargs='+', 
                        help='The workloads that the models experimented.')
    parser.add_argument('-m', "--mctrace", action='store_const', const=True,
                        default=False, help="The input traces from shell are for multi-core systems.")
    parser.add_argument('-r', "--results", type=str,
                        default="./results_200M", help="The dirctory for results.")
    args = parser.parse_args()
    if args.model and args.workloads and args.results:
        workloads = getattr(utils, args.workloads[0], None)
        if not workloads:
            # record results of a single workload
            workloads = args.workloads
            workloads = [workloads] if args.mctrace else workloads
        insert_or_overwrite_results(get_collection(), [args.model], workloads, args.results)
        print(f"Insert or update db for model {args.model} on workloads {args.workloads}, Succeed!")


if __name__ == "__main__":
    shell_call()
    # insert_or_overwrite_results(get_collection(), 'bimodal-no-pythia-no-no-lru-4core', utils.trace_mix2_4core, "./results_4core_200M")
