#!/usr/bin/python3

from multiprocessing.pool import AsyncResult
import subprocess
import time
from multiprocessing.context import Process
from multiprocessing.managers import BaseManager
from multiprocessing import Queue, Pool
from collections import defaultdict


class TaskManager(BaseManager):
    pass

queue = Queue()
TaskManager.register("get_queue", callable=lambda:queue)

def start_server():
    manager = TaskManager(address=('', 50000), authkey=b'jsz1995')
    server = manager.get_server()
    server.serve_forever()

def start_worker():
    manager = TaskManager(address=('', 50000), authkey=b'jsz1995')
    manager.connect()
    work_queue = manager.get_queue()
    with Pool(15) as pool: 
        while True:
            print("waiting for work")
            work = work_queue.get()
            print(work)
            pool.apply_async(subprocess.run, args=(work,), kwds={"shell":True})

if __name__ == "__main__":
    Process(target=start_server).start()
    time.sleep(0.1)
    start_worker()

