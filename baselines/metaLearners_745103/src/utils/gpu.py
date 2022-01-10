# this tool is based on the output of os.system('nvidia-smi')
import os

class GPU:
    def __init__(self, idx, uuid, mem_total, mem_used, gputil) -> None:
        super().__init__()
        self.idx = idx
        self.uuid = uuid
        self.mem_total = mem_total
        self.mem_used = mem_used
        self.mem_remain = mem_total - mem_used
        self.gputil = gputil

    @classmethod
    def auto_create(cls):
        output = os.popen('nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader')
        lines = [[y.strip() for y in x.strip().split(',')] for x in output.readlines()]
        gpus = []
        for l in lines:
            gpus.append(GPU(int(l[0]), l[1], int(l[2].split()[0]), int(l[3].split()[0]), int(l[4].split()[0])))
        return gpus
        
class GPUManager:
    def __init__(self) -> None:
        # initialize gpu id mapping
        self.gpu_bus_to_id = {}
        output = os.popen('nvidia-smi --query-gpu=index,uuid --format=csv,noheader')
        output = output.readlines()
        lines = [x.strip().split(',') for x in output]
        for line in lines:
            self.gpu_bus_to_id[line[1].strip()] = int(line[0].strip())
        self.history = None
    
    def query_processes(self, ppid=None):
        output = os.popen('nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader')
        lines = [[y.strip() for y in x.strip().split(',')] for x in output.readlines()]
        
        for i in range(len(lines)):
            uuid, pid, process_name, memory = lines[i]
            gpuid = self.gpu_bus_to_id[uuid]
            lines[i] = {
                'gpuid': gpuid,
                'pid': int(pid),
                'process_name': process_name,
                'memory': int(memory.split()[0])
            }
    
        if ppid is None:
            return lines
        
        return [x for x in lines if x['pid'] == ppid]

    def print_processes(self, ppid=None, prin=print):
        processes = self.query_processes(ppid)
        if self.history == processes:
            return
        else:
            self.history = processes
            prin(processes)

    def get_gpus(self, remain_memory=5000):
        gpus = GPU.auto_create()
        gpus = [x for x in gpus if x.mem_remain >= remain_memory]
        return sorted(gpus, key=lambda x:x.mem_remain + x.idx * 0.1, reverse=True)