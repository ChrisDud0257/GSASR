import os
import io
import sys
import timeit
import tokenize
import torch
import psutil
import inspect
from loguru import logger
from prettytable import PrettyTable

# implement by xtudbxk
# github: https://github.com/xtudbxk/lineprofiler
class MyLineProfiler():
    def __init__(self, base='ms', cuda_sync=True, gpuids=(0,), warmup=0, warmup_lineno=-1):

        if base == 'ms':
            self.base_n = 1000
        elif base == 's':
            self.base_n = 1
        else:
            logguru.warning(f'Unsupported base - {base}, using "s" instead')

        self.base = base
        self.cuda_sync = cuda_sync
        self.gpuids = gpuids
        self.warmup = warmup
        self.warmup_counter = warmup
        # we should wait this line execute warup_counter times 
        # before recording the stats
        self.warmup_lineno = warmup_lineno

        # for time profiling
        self._times = {}
        self._func_name = None
        self._func_filename = None
        self._last_time = -1
        self._last_lineno = -1
        self._func_hit_count = 0
        self._func_firstlineno = 0

        # for memory profiling
        self._process = psutil.Process(os.getpid())
        self._memory = {}
        self._last_memory = 0

        # for cuda memory profiling
        self._gpu_memory = {}
        self._gpu_last_memory = 0

    def __trace_func__(self, frame, event, arg):
        # print(f'in {frame.f_code.co_filename} func {frame.f_code.co_name} line {frame.f_lineno}, event - {event}')

        # check if run into the decorated func
        if self._func_firstlineno == frame.f_code.co_firstlineno and frame.f_code.co_name == self._func_name and frame.f_code.co_filename == self._func_filename:

            # --- obtain info for current hit ---
            # cuda related
            if self.cuda_sync is True:
                torch.cuda.synchronize()

            current_time = timeit.default_timer()
            memory = self._process.memory_info().rss
            gpu_memory = torch.cuda.memory_allocated()
            # --- ends ---
            
            # --- initilize the info when first hit ---
            if frame.f_lineno not in self._times: # first hit time for this line
                self._times[frame.f_lineno] = {'hit':0, 'time': 0}
                self._memory[frame.f_lineno] = 0
                self._gpu_memory[frame.f_lineno] = 0
            # --- ends ---

            # --- record info before call the decorated func ---
            # 'call' - before call the func
            if event == 'call':
                self._last_time = current_time
                self._last_lineno = frame.f_lineno
                self._last_memory = memory
                self._last_gpu_memory = gpu_memory

                if self.warmup_lineno < 0:
                    self.warmup_counter -= 1
                    if self.warmup_counter < 0:
                        self._func_hit_count += 1
            # --- ends ---

            # 'line' - after excuting the line
            # 'return' - return from the function
            if event == 'line' or event == 'return':

                if event == 'line' and self.warmup_counter < 0:
                    self._times[frame.f_lineno]['hit'] += 1


                # --- obtain the memory and time consumed by this line ---
                if self.warmup_counter < 0:
                    self._times[self._last_lineno]['time'] += current_time - self._last_time
                self._memory[self._last_lineno] += memory - self._last_memory
                self._gpu_memory[self._last_lineno] += gpu_memory - self._gpu_last_memory
                # --- ends ---

                if self.cuda_sync is True:
                    torch.cuda.synchronize()

                self._last_time = timeit.default_timer()
                self._last_memory = memory
                self._gpu_last_memory = gpu_memory
                self._last_lineno = frame.f_lineno

        return self.__trace_func__

    def decorate(self, func):
        if self._func_name is not None:
            logger.warning(f'Only support decorate only one func. Aready decorated "{self._func_name}"')
        self._func_name = func.__name__
        self._func_filename = func.__code__.co_filename
        self._func_firstlineno = func.__code__.co_firstlineno

        def _f(*args, **kwargs):
            origin_trace_func = sys.gettrace()
            sys.settrace(self.__trace_func__)
            ret = func(*args, **kwargs)
            sys.settrace(origin_trace_func)
            return ret
        return _f

    def _get_table(self):

        if len(self._times) <= 0:
            logger.warning(f"un recorded datas, please ensure the function is executed")
            return None
        
        # --- load the source code ---
        with open(self._func_filename, 'r') as f:
            source_lines = [line.strip('\n') for line in f.readlines()]
            code_str = "\n".join(source_lines)

        def_lineno = min(self._times.keys())
        final_lineno = max(self._times.keys())

        # remove the additional blank content
        pre_blank_count = len(source_lines[def_lineno-1]) - len(source_lines[def_lineno-1].lstrip(' ').lstrip('\t'))
        # --- ends ---

        # --- analysize the source code and collect infos for multi-line code ---
        new_logic_linenos = [token.start[0] for token in tokenize.generate_tokens(
            io.StringIO(code_str).readline) if token.type == 4]
        # --- ends ---

        # --- merge the stats multi-line code ---
        sorted_linenos = [lineno for lineno in self._times.keys()]
        sorted_linenos.sort(key=int)
        
        lineno_cache = []
        for lineno in sorted_linenos:
            if lineno not in new_logic_linenos: 
                lineno_cache.append(lineno)
            else:
                # we should merge its info to the prev_lineno
                if len(lineno_cache) <= 0:
                    continue
                else:
                    lineno_cache.append(lineno)
                    first_lineno = lineno_cache[0]
                    for prev_lineno in lineno_cache[1:]:
                        self._times[first_lineno]["hit"] = min(self._times[first_lineno]["hit"], self._times[prev_lineno]["hit"])
                        self._times[first_lineno]["time"] += self._times[prev_lineno]["time"]
                        del self._times[prev_lineno]

                        self._memory[first_lineno] += self._memory[prev_lineno]
                        del self._memory[prev_lineno]

                        self._gpu_memory[first_lineno] += self._gpu_memory[prev_lineno]
                        del self._gpu_memory[prev_lineno]
                    lineno_cache = []
        # --- ends ---

        # --- initialize the pretty table for output ---
        table = PrettyTable(['lineno', 'hits', 'time', 'time per hit', 'hit perc', 'time perc', 'mem inc', 'mem peak', 'gpu mem inc', 'gpu mem peak'])
        # --- ends ---

        # --- compute some statisticals ---
        total_hit = 0 # for compute the hit percentage
        total_time = 0
        for lineno, stats in self._times.items():
            if lineno == def_lineno: continue
            total_hit += stats['hit']
            total_time += stats['time']

        total_memory = sum([m for l,m in self._memory.items()]) / 1024 / 1024 
        total_gpu_memory = sum([m for l,m in self._gpu_memory.items()]) / 1024 / 1024
        # --- ends ---

        peak_cpu_memory = 0
        peak_gpu_memory = 0
        for lineno in range(def_lineno, final_lineno+1):
            if lineno not in self._times:
                # the comment line, empty line or merged line from multi-lines code
                table.add_row([lineno, '-', '-', '-', '-', '-', '-',f'{peak_cpu_memory:5.3f} MB', '-', f'{peak_gpu_memory:5.3f} MB'])
            else:
                stats = self._times[lineno]
                if lineno == def_lineno: 
                    table.add_row([lineno, self._func_hit_count, f'{total_time*self.base_n:.4f} {self.base}', f'{total_time/self._func_hit_count*self.base_n:.4f} {self.base}', '-', '-', f'{total_memory:5.3f} MB', 'baseline', f'{total_gpu_memory:5.3f} MB', 'baseline'])
                else:

                    line_result = [lineno, stats['hit'], 
                                  f'{stats["time"]*self.base_n:.4f} {self.base}', 
                                  f'{stats["time"]/stats["hit"]*self.base_n:.4f} {self.base}' if stats['hit'] > 0 else 'nan', 
                                  f'{stats["hit"]/total_hit*100:.3f}%' if total_hit > 0 else 'nan', 
                                  f'{stats["time"]/total_time*100:.3f}%'] if total_time > 0 else 'nan'

                    line_result += [f'{self._memory[lineno]/1024/1024:5.3f} MB' if stats['hit'] > 0 else '0 MB']
                    peak_cpu_memory = peak_cpu_memory + self._memory[lineno]/1024/1024
                    line_result += [f'{peak_cpu_memory:5.3f} MB']

                    line_result += [f'{self._gpu_memory[lineno]/1024/1024:5.3f} MB' if stats['hit'] > 0 else '0 MB']
                    peak_gpu_memory = peak_gpu_memory + self._gpu_memory[lineno]/1024/1024
                    line_result += [f'{peak_gpu_memory:5.3f} MB']

                    table.add_row(line_result)

        table.add_column('sources', [source_lines[i-1][pre_blank_count:] if len(source_lines[i-1])>pre_blank_count else '' for i in range(def_lineno, final_lineno+1)], 'l')
        return table

    def print(self, filename=None, mode="w"):
        introducation = '''
1. The first line of table reports the overall results of the whole function and the following lines reports the statistics of each line in the function.
2. The `hit perc` and `time perc` represent `hit percentage` and `time percentage`.
3. For memory, there exists four categories `mem inc`, `mem peak`, `gpu mem inc` and `gpu mem peak`. They denotes `cpu memory increasement`, `cpu memory peak`, `gpu memory increasement` and `gpu memory peak`. All the results are collected in the last run. The number in the increasement field denots the increasement of corresponding memory of each line (the first line is related to the whole function). Sometimes, the number of each line is far less of the number of the first line, which is valid since python may auto release the unused memory after the function execution. The number of each line in the peak filed is a simple sum of the numbers of above lines in the increasement field, which is used to demonstrate the possible maxinum memory usage in the function.
4. For any issue, please concact us via https://github.com/xtudbxk/lineprofiler or zhengqiang.zhang@hotmail.com
        '''
        print(introducation)

        table = PrettyTable(['lineno', 'hits', 'time', 'time per hit', 'hit perc', 'time perc', 'mem inc', 'mem peak', 'gpu mem inc', 'gpu mem peak'])
        table = self._get_table()
        print(table)
        if filename is not None:
            with open(filename, mode) as f:
                f.write(introducation)
                f.write(f"args - base={self.base}, cuda_sync={self.cuda_sync}, gpuids={self.gpuids}, warmup={self.warmup}\n")
                f.write(str(table))
            
if __name__ == '__main__':
    import numpy as np
    def mytest(h='hello', 
               xx="xx"):
    
        h = h + 'world'
        a = []
        for _ in range(200):
            # a = np.zeros((1000, 1000), dtype=np.float32)
            a.append(np.zeros((1000, 1000), dtype=np.float32))
            a.append(
                    np.zeros((1000, 1000), 
                              dtype=np.float32))
            # print(a[0,0])
        print(h)

    profiler = MyLineProfiler(cuda_sync=False, warmup=2)
    mytest = profiler.decorate(mytest)
    for _ in range(5):
        mytest()
    profiler.print()
