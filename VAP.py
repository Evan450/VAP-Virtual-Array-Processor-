#!/usr/bin/env python3
"""
 Copyright (C) 2025 Discover Interactive

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see https://choosealicense.com/licenses/gpl-3.0/ or https://www.gnu.org/licenses/gpl-3.0.en.html.

-- Updates and Secondary Header --

Name: Virtual Array Processor (VAP)
Author: Discover Interactive
Version: 5.4b
Description:
  - Removed 'cp.errstate' usage to avoid AttributeError on older CuPy versions
  - Added Manual clamps for large GPU results
  - Added emergency try/except blocks for critical errors
"""

import os
import sys
import math
import random
import time
import asyncio
import logging
from logging import handlers
import json
import yaml
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import psutil

try:
    import aiofiles
except ImportError:
    aiofiles = None

# GPU libraries
try:
    import cupy as cp
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# =============================================================================
# 1) Configuration
# =============================================================================
@dataclass
class VAPConfig:
    number_of_tasks: int = 50_000
    tasks_per_chunk: int = 2_000
    compute_mode: str = "CPU+GPU"   # "CPU", "GPU", "CPU+GPU"
    output_file: Optional[str] = "Output.txt"
    task_complexity_mode: str = "mixed"  # "simple", "mixed", "complex"
    log_dir: str = "logs"
    cache_dir: str = "cache"
    checkpoint_interval: int = 10
    monitoring_interval: float = 1.0
    max_retries: int = 3
    gpu_memory_limit: float = 0.9
    max_concurrent_chunks: int = 2

    def __post_init__(self):
        if self.number_of_tasks < 1:
            raise ValueError("Number of Tasks must be positive.")
        if self.tasks_per_chunk < 1:
            raise ValueError("Tasks Per Chunk must be positive.")
        if self.compute_mode not in ["CPU", "GPU", "CPU+GPU"]:
            raise ValueError("Invalid compute_mode.")
        if self.task_complexity_mode not in ["simple", "mixed", "complex"]:
            raise ValueError("Invalid task_complexity_mode.")

class ConfigManager:
    DEFAULT_CONFIG_PATH = "vap_config.yaml"
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "VAPConfig":
        path = config_path or cls.DEFAULT_CONFIG_PATH
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f)
                    return VAPConfig(**data)
                except Exception as e:
                    logging.warning(f"Error loading config: {e}. Using defaults.")
                    return VAPConfig()
        return VAPConfig()
    
    @classmethod
    def save(cls, config: "VAPConfig", config_path: Optional[str] = None):
        path = config_path or cls.DEFAULT_CONFIG_PATH
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(asdict(config), f)

# =============================================================================
# 2) Logging Setup
# =============================================================================
class LogManager:
    def __init__(self, config: VAPConfig):
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger("VAP")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        err_handler = handlers.RotatingFileHandler(
            self.log_dir / "vap_error.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        err_handler.setLevel(logging.ERROR)
        
        info_handler = handlers.RotatingFileHandler(
            self.log_dir / "vap_info.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        info_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        for h in [err_handler, info_handler, console_handler]:
            h.setFormatter(formatter)
            self.logger.addHandler(h)

# =============================================================================
# 3) Task Management
# =============================================================================
@dataclass
class Task:
    op: str
    x: float
    y: Optional[float]
    priority: int = 0
    id: str = ""
    def __post_init__(self):
        if not self.id:
            import random
            self.id = f"{self.op}_{self.x}_{self.y}_{random.randint(0,999999)}"

class TaskCache:
    def __init__(self, config: VAPConfig, logger: logging.Logger):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._memory_cache = {}
        self.logger = logger
        
    async def aget(self, task_id: str) -> Optional[float]:
        if task_id in self._memory_cache:
            return self._memory_cache[task_id]
        cache_file = self.cache_dir / f"{task_id}.json"
        if aiofiles and cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    data = json.loads(await f.read())
                    self._memory_cache[task_id] = data['result']
                    return data['result']
            except (json.JSONDecodeError, KeyError, IOError) as e:
                self.logger.warning(f"Cache read error: {e}")
        return None
        
    async def aset(self, task_id: str, result: float):
        self._memory_cache[task_id] = result
        cache_file = self.cache_dir / f"{task_id}.json"
        if aiofiles:
            try:
                async with aiofiles.open(cache_file, 'w') as f:
                    await f.write(json.dumps({'task_id': task_id, 'result': result}))
            except IOError as e:
                self.logger.error(f"Cache write error: {e}")

class Checkpointer:
    def __init__(self, config: VAPConfig, logger: logging.Logger):
        self.config = config
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"
    
    async def save_incremental(self, tasks: List[Task], chunk_index: int):
        if not aiofiles:
            return
        try:
            existing_data = {}
            if self.checkpoint_file.exists():
                async with aiofiles.open(self.checkpoint_file, 'r') as f:
                    try:
                        existing_data = json.loads(await f.read())
                    except json.JSONDecodeError:
                        existing_data = {}
            if "processed_chunks" not in existing_data:
                existing_data["processed_chunks"] = {}
            
            existing_data["processed_chunks"][str(chunk_index)] = [asdict(t) for t in tasks]
            
            async with aiofiles.open(self.checkpoint_file, 'w') as f:
                await f.write(json.dumps(existing_data, indent=2))
            
            self.logger.info(f"Checkpoint saved for chunk {chunk_index}.")
        except IOError as e:
            self.logger.error(f"Checkpoint save error: {e}")
    
    async def load_all(self) -> Optional[Dict[str, Any]]:
        if not aiofiles or not self.checkpoint_file.exists():
            return None
        try:
            async with aiofiles.open(self.checkpoint_file, 'r') as f:
                data = json.loads(await f.read())
                self.logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
                return data
        except Exception as e:
            self.logger.error(f"Checkpoint load error: {e}")
            return None

class TaskManager:
    def __init__(self, config: VAPConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.tasks: List[Task] = []
        self.cache = TaskCache(config, logger)
        self.checkpointer = Checkpointer(config, logger)
    
    def generate_tasks(self, n: Optional[int] = None) -> List[Task]:
        simple_ops  = ['add', 'sub', 'mul', 'div', 'mod', 'pow']
        complex_ops = ['exp', 'sqrt', 'log', 'sin', 'cos', 'tan']
        total = n or self.config.number_of_tasks

        tasks = []
        for _ in range(total):
            mode = self.config.task_complexity_mode
            if mode == "simple":
                op = random.choice(simple_ops)
            elif mode == "complex":
                op = random.choice(complex_ops)
            else:  # "mixed"
                op = random.choice(simple_ops + complex_ops)

            if op in complex_ops:
                # Bound inputs so log and exp don't overflow (1..300).
                x = random.uniform(1, 300)
                y = None
            else:
                x = random.uniform(1, 1_000_000)
                y = random.uniform(1, 1_000_000)
                if op == 'pow':
                    # Avoid catastrophic overflow
                    x = random.uniform(1, 200)
                    y = random.uniform(1, 5)

            tasks.append(Task(op=op, x=x, y=y))
        
        self.logger.info(f"Generated {len(tasks)} tasks (mode: {mode}).")
        return tasks
    
    async def add_task(self, task: Task):
        cached = await self.cache.aget(task.id)
        if cached is None:
            self.tasks.append(task)
        else:
            self.logger.info(f"Task {task.id} already cached; skipping.")
    
    async def restore(self) -> bool:
        data = await self.checkpointer.load_all()
        if data and "processed_chunks" in data:
            self.logger.info(f"Loaded checkpoint with {len(data['processed_chunks'])} chunk(s).")
            return True
        return False

# =============================================================================
# 4) Resource Monitoring
# =============================================================================
class ResourceMonitor:
    def __init__(self, config: VAPConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.is_monitoring = False
        self.metrics = {}
        self._lock = threading.Lock()

    def start(self):
        self.is_monitoring = True
        self.metrics = {}
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started.")

    def stop(self):
        self.is_monitoring = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join()
        summary = self.get_summary()
        self._print_summary(summary)
        self.logger.info("Resource monitoring stopped.")
        return summary

    def _monitor_loop(self):
        from collections import defaultdict
        self.metrics = defaultdict(list)
        while self.is_monitoring:
            try:
                with self._lock:
                    cpu_p = psutil.cpu_percent()
                    mem = psutil.virtual_memory()
                    self.metrics['cpu'].append(cpu_p)
                    self.metrics['memory'].append(mem.percent)
                    if GPU_AVAILABLE:
                        try:
                            gpus = GPUtil.getGPUs()
                            for gpu in gpus:
                                self.metrics[f'gpu_{gpu.id}_util'].append(gpu.load * 100)
                                self.metrics[f'gpu_{gpu.id}_mem'].append(gpu.memoryUtil * 100)
                        except Exception as e:
                            self.logger.warning(f"GPU monitoring error: {e}")
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
            time.sleep(self.config.monitoring_interval)

    def get_summary(self) -> Dict[str,Dict[str,float]]:
        from statistics import mean, stdev
        with self._lock:
            summary = {}
            for metric, values in self.metrics.items():
                if values:
                    summary[metric] = {
                        'mean': float(mean(values)),
                        'max': float(max(values)),
                        'min': float(min(values)),
                        'std': float(stdev(values)) if len(values)>1 else 0.0
                    }
            return summary

    def _print_summary(self, summary: Dict[str,Dict[str,float]]):
        print("\n=== Resource Usage Summary ===")
        print(f"{'Metric':<15}{'Mean':>10}{'Max':>10}{'Min':>10}{'Std Dev':>10}")
        print("-"*55)
        for metric, stats in summary.items():
            print(f"{metric:<15}{stats['mean']:>10.2f}{stats['max']:>10.2f}{stats['min']:>10.2f}{stats['std']:>10.2f}")

# =============================================================================
# 5) Progress Tracker
# =============================================================================
class ProgressTracker:
    def __init__(self, total_tasks: int, logger: logging.Logger):
        self.total_tasks = total_tasks
        self.tasks_done = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
        self.logger = logger

    def update(self, new_done: int):
        with self._lock:
            self.tasks_done = new_done
            self.display()

    def display(self):
        elapsed = time.time() - self.start_time
        fraction = self.tasks_done / self.total_tasks if self.total_tasks else 0
        eta = (elapsed / fraction - elapsed) if fraction>0 else 0
        print(f"[Progress] {self.tasks_done}/{self.total_tasks} tasks | Elapsed: {elapsed:.2f}s | ETA: {eta:.2f}s", end='\r')

# =============================================================================
# 6) Arithmetic Operations
# =============================================================================
CPU_OPS = {
    'add':0,'sub':1,'mul':2,'div':3,'mod':4,'pow':5,
    'exp':6,'sqrt':7,'log':8,'sin':9,'cos':10,'tan':11
}
GPU_SIMPLE_OPS = {
    'add':0,'sub':1,'mul':2,'div':3,'mod':4,'pow':5
}

def cpu_vectorized(ops: np.ndarray, xvals: np.ndarray, yvals: np.ndarray) -> np.ndarray:
    results = np.empty_like(xvals, dtype=np.float64)
    for i in range(len(ops)):
        opcode = ops[i]
        x = xvals[i]
        y = yvals[i]
        val = math.nan
        try:
            if opcode == 0:   # add
                val = x + y
            elif opcode == 1: # sub
                val = x - y
            elif opcode == 2: # mul
                val = x * y
            elif opcode == 3: # div
                val = x / y if y != 0 else math.nan
            elif opcode == 4: # mod
                val = x % y if y != 0 else math.nan
            elif opcode == 5: # pow
                val = math.pow(x, y)
            elif opcode == 6: # exp
                val = math.exp(x)
            elif opcode == 7: # sqrt
                val = math.sqrt(x) if x >= 0 else math.nan
            elif opcode == 8: # log
                val = math.log(x) if x>0 else math.nan
            elif opcode == 9: # sin
                val = math.sin(x)
            elif opcode == 10: # cos
                val = math.cos(x)
            elif opcode == 11: # tan
                val = math.tan(x)
        except OverflowError:
            val = float('inf')
        except ValueError:
            val = math.nan
        results[i] = val
    return results

def gpu_vectorized(op_array: "cp.ndarray", x_array: "cp.ndarray", y_array: "cp.ndarray") -> "cp.ndarray":
    """
    GPU-based vectorization for simple ops. 
    We do NOT use cp.errstate here. We'll clamp after the kernel if needed.
    """
    out = cp.zeros_like(x_array, dtype=cp.float32)
    add_mask = (op_array == 0)
    sub_mask = (op_array == 1)
    mul_mask = (op_array == 2)
    div_mask = (op_array == 3)
    mod_mask = (op_array == 4)
    pow_mask = (op_array == 5)

    out[add_mask] = x_array[add_mask] + y_array[add_mask]
    out[sub_mask] = x_array[sub_mask] - y_array[sub_mask]
    out[mul_mask] = x_array[mul_mask] * y_array[mul_mask]

    div_nonzero = div_mask & (y_array != 0)
    out[div_nonzero] = x_array[div_nonzero] / y_array[div_nonzero]
    out[div_mask & (y_array == 0)] = cp.nan

    mod_nonzero = mod_mask & (y_array != 0)
    out[mod_nonzero] = cp.mod(x_array[mod_nonzero], y_array[mod_nonzero])
    out[mod_mask & (y_array == 0)] = cp.nan

    out[pow_mask] = cp.power(x_array[pow_mask], y_array[pow_mask])
    return out

def process_gpu_tasks(task_tuples: List[Tuple[str,float,Optional[float]]], logger: logging.Logger) -> List[float]:
    """
    GPU processing for simple ops, no cp.errstate, with manual post-clamp for large values.
    """
    if not GPU_AVAILABLE:
        logger.error("GPU not available but GPU function called.")
        return [math.nan]*len(task_tuples)

    try:
        codes = []
        xs = []
        ys = []
        for op,x,y in task_tuples:
            if op in GPU_SIMPLE_OPS:
                codes.append(GPU_SIMPLE_OPS[op])
                xs.append(x)
                ys.append(y if y is not None else 1.0)
            else:
                codes.append(-1)
                xs.append(float('nan'))
                ys.append(float('nan'))

        op_cp = cp.array(codes, dtype=cp.int32)
        x_cp  = cp.array(xs,    dtype=cp.float32)
        y_cp  = cp.array(ys,    dtype=cp.float32)

        out_cp = gpu_vectorized(op_cp, x_cp, y_cp)

        out_np = out_cp.get().astype(float)

        # Post-processing clamp for very large values => inf
        # e.g. threshold ~1e300 for float64
        HUGE_THRESHOLD = 1e300
        out_np[np.abs(out_np) > HUGE_THRESHOLD] = float('inf')

        return out_np.tolist()

    except Exception as e:
        logger.exception(f"GPU error: {e}")
        return [math.nan]*len(task_tuples)

# =============================================================================
# 7) Processing Engine
# =============================================================================
class ProcessingEngine:
    def __init__(self, config: VAPConfig, logger: logging.Logger, task_manager: TaskManager):
        self.config = config
        self.logger = logger
        self.task_manager = task_manager
        self.resource_monitor = ResourceMonitor(config, logger)
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_chunks)
        self.progress_tracker: Optional[ProgressTracker] = None

    async def process_all(self, tasks: List[Task]) -> List[float]:
        """
        High-level emergency try/except in case a catastrophic error
        happens inside the entire pipeline. We'll attempt to gracefully
        stop if something truly unrecoverable occurs.
        """
        try:
            self.resource_monitor.start()
            self.progress_tracker = ProgressTracker(len(tasks), self.logger)
            chunks = self._create_chunks(tasks)
            self.logger.info(f"Processing {len(tasks)} tasks in {len(chunks)} chunk(s).")

            coros = [self._process_chunk_wrapper(ch, i+1) for i,ch in enumerate(chunks)]
            chunk_results_list = await asyncio.gather(*coros)

            results = []
            for cr in chunk_results_list:
                results.extend(cr)
            return results

        except Exception as e:
            # Emergency: unknown catastrophic error
            self.logger.exception(f"CRITICAL ERROR in process_all: {e}")
            print("\nA critical error occurred during processing. The script will halt gracefully.")
            # Attempt a safe stop
            self.resource_monitor.stop()
            # Return empty or partial results
            return []

        finally:
            self.resource_monitor.stop()

    def _create_chunks(self, tasks: List[Task]) -> List[List[Task]]:
        sz = self._calc_chunk_size()
        return [tasks[i:i+sz] for i in range(0, len(tasks), sz)]

    def _calc_chunk_size(self) -> int:
        base_size = self.config.tasks_per_chunk
        try:
            mem = psutil.virtual_memory()
            mem_gb = mem.available / (1024**3)
            mem_factor = max(0.5, min(2.0, mem_gb/8.0))
            base_size = int(base_size * mem_factor)

            cpu_count = psutil.cpu_count(logical=True)
            cpu_factor = max(0.5, min(2.0, cpu_count/4.0))
            base_size = int(base_size * cpu_factor)

            if GPU_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    g0 = gpus[0]
                    avail_mem = g0.memoryTotal * self.config.gpu_memory_limit
                    gpu_gb = avail_mem / 1024
                    gpu_factor = max(0.5, min(2.0, gpu_gb/4.0))
                    base_size = int(base_size * gpu_factor)
        except Exception as e:
            self.logger.warning(f"Chunk size calc error: {e}")

        return max(1, base_size)

    async def _process_chunk_wrapper(self, chunk: List[Task], idx: int) -> List[float]:
        async with self.semaphore:
            try:
                results = await self._process_chunk(chunk)
                await self.task_manager.checkpointer.save_incremental(chunk, idx)
                return results
            except Exception as e:
                self.logger.error(f"EMERGENCY: chunk {idx} crashed with error: {e}")
                print(f"\nChunk {idx} encountered a critical error: {e}")
                print("Unable to recover from this chunk—marking all tasks as NaN for this chunk.")
                return [math.nan]*len(chunk)

    async def _process_chunk(self, chunk: List[Task]) -> List[float]:
        for attempt in range(1, self.config.max_retries+1):
            try:
                mode = self.config.compute_mode
                if mode == "GPU" and GPU_AVAILABLE:
                    out = await self._gpu_only(chunk)
                elif mode == "CPU+GPU" and GPU_AVAILABLE:
                    out = await self._hybrid(chunk)
                else:
                    out = await self._cpu_only(chunk)
                
                new_done = self.progress_tracker.tasks_done + len(chunk)
                self.progress_tracker.update(new_done)
                return out

            except Exception as e:
                self.logger.error(f"Chunk attempt {attempt} failed: {e}")
                if attempt < self.config.max_retries:
                    self.logger.info("Retrying chunk...")
                    await asyncio.sleep(1)
                else:
                    self.logger.error("Max retries reached; returning NaNs.")
                    return [math.nan]*len(chunk)

    async def _cpu_only(self, chunk: List[Task]) -> List[float]:
        loop = asyncio.get_running_loop()
        arr = [(t.op, t.x, t.y if t.y else 0.0) for t in chunk]

        def do_cpu():
            ops = np.array([CPU_OPS[a[0]] for a in arr], dtype=np.int32)
            xs  = np.array([a[1] for a in arr], dtype=np.float64)
            ys  = np.array([a[2] for a in arr], dtype=np.float64)
            r   = cpu_vectorized(ops, xs, ys)
            return r.tolist()

        with concurrent.futures.ThreadPoolExecutor() as exec_:
            results = await loop.run_in_executor(exec_, do_cpu)
        await self._cache_results(chunk, results)
        return results

    async def _gpu_only(self, chunk: List[Task]) -> List[float]:
        loop = asyncio.get_running_loop()
        arr = [(t.op, t.x, t.y) for t in chunk]
        # Additional try/except if GPU is critical
        try:
            results = await loop.run_in_executor(None, process_gpu_tasks, arr, self.logger)
        except Exception as ex:
            self.logger.error(f"CRITICAL GPU error: {ex}")
            print(f"\nCRITICAL GPU error: {ex}. Halting the script gracefully.")
            sys.exit(1)  # or raise, if you want to bubble up
        await self._cache_results(chunk, results)
        return results

    async def _hybrid(self, chunk: List[Task]) -> List[float]:
        simple = [t for t in chunk if t.op in GPU_SIMPLE_OPS]
        complex_ = [t for t in chunk if t.op not in GPU_SIMPLE_OPS]
        gpu_future = asyncio.create_task(self._gpu_only(simple))
        cpu_future = asyncio.create_task(self._cpu_only(complex_))

        # If GPU fails catastrophically, we do an emergency exit
        gpu_res, cpu_res = await asyncio.gather(gpu_future, cpu_future)
        out = []
        g_idx, c_idx = 0, 0
        for t in chunk:
            if t.op in GPU_SIMPLE_OPS:
                out.append(gpu_res[g_idx])
                g_idx += 1
            else:
                out.append(cpu_res[c_idx])
                c_idx += 1
        return out

    async def _cache_results(self, chunk: List[Task], results: List[float]):
        for t,r in zip(chunk, results):
            await self.task_manager.cache.aset(t.id, r)

# =============================================================================
# 8) Controller + Menus
# =============================================================================
class VAPController:
    def __init__(self, config: VAPConfig, tm: TaskManager, engine: ProcessingEngine):
        self.config = config
        self.task_manager = tm
        self.engine = engine

    def set_compute_mode(self, code: int) -> bool:
        if code == 1:
            self.config.compute_mode = "CPU"
        elif code == 2:
            if not GPU_AVAILABLE:
                self.engine.logger.warning("GPU not available.")
                return False
            self.config.compute_mode = "GPU"
        elif code == 3:
            if not GPU_AVAILABLE:
                self.engine.logger.warning("GPU not available.")
                return False
            self.config.compute_mode = "CPU+GPU"
        else:
            return False
        return True

    def set_number_of_tasks(self, val: int) -> bool:
        if val < 1:
            return False
        self.config.number_of_tasks = val
        return True

    def set_tasks_per_chunk(self, val: int) -> bool:
        if val < 1:
            return False
        self.config.tasks_per_chunk = val
        return True

    def set_output_file(self, fname: str) -> bool:
        self.config.output_file = fname or None
        return True

    def set_task_complexity_mode(self, code: int) -> bool:
        if code == 1:
            self.config.task_complexity_mode = "simple"
        elif code == 2:
            self.config.task_complexity_mode = "mixed"
        elif code == 3:
            self.config.task_complexity_mode = "complex"
        else:
            return False
        return True

    def get_settings(self) -> Dict[str,Any]:
        return asdict(self.config)

    def generate_new_tasks(self) -> bool:
        new_ts = self.task_manager.generate_tasks()
        self.task_manager.tasks = new_ts
        return True

    def clear_tasks(self) -> bool:
        self.task_manager.tasks.clear()
        return True

    async def add_task(self, op: str, x: float, y: Optional[float]=None) -> bool:
        if op not in CPU_OPS:
            self.engine.logger.warning(f"Invalid op: {op}")
            return False
        t = Task(op=op, x=x, y=y)
        await self.task_manager.add_task(t)
        return True

    def remove_task(self, idx: int) -> bool:
        if 1 <= idx <= len(self.task_manager.tasks):
            self.task_manager.tasks.pop(idx-1)
            return True
        return False

    def modify_task(self, idx: int, new_op: Optional[str],
                    new_x: Optional[float], new_y: Optional[float]) -> bool:
        if not (1 <= idx <= len(self.task_manager.tasks)):
            return False
        t = self.task_manager.tasks[idx-1]
        if new_op and new_op in CPU_OPS:
            t.op = new_op
        if new_x is not None:
            t.x = new_x
        if new_y is not None:
            t.y = new_y
        return True

    def get_tasks(self) -> List[Task]:
        return list(self.task_manager.tasks)

    async def run_computation(self) -> List[float]:
        """
        Another emergency try/except here if something major
        occurs in the main computation pipeline.
        """
        try:
            if not self.task_manager.tasks:
                self.engine.logger.info("No tasks available.")
                return []
            results = await self.engine.process_all(self.task_manager.tasks)
            if self.config.output_file:
                try:
                    with open(self.config.output_file, "w", encoding="utf-8") as f:
                        f.write("Index,Operation,X,Y,Result\n")
                        for i, (task, r) in enumerate(zip(self.task_manager.tasks, results), start=1):
                            ystr = str(task.y) if task.y is not None else ""
                            f.write(f"{i},{task.op},{task.x},{ystr},{r}\n")
                    self.engine.logger.info(f"Results written to {self.config.output_file}")
                except Exception as e:
                    self.engine.logger.error(f"File write error: {e}")
            return results
        except Exception as e:
            self.engine.logger.exception(f"CRITICAL ERROR in run_computation: {e}")
            print("\nA critical error occurred during run_computation. Halting gracefully.")
            return []

def main_menu(controller: VAPController):
    while True:
        clear_screen()
        tasks_count = len(controller.get_tasks())
        print("===== Virtual Array Processor =====")
        print(f"[Tasks in memory: {tasks_count}]")
        print("1) Start Computation")
        print("2) Settings")
        print("3) Manage Tasks")
        print("4) Exit")
        choice = input("Choice: ").strip()
        if choice == "1":
            asyncio.run(do_computation(controller))
        elif choice == "2":
            settings_menu(controller)
        elif choice == "3":
            manage_tasks_menu(controller)
        elif choice == "4":
            clear_screen()
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid choice.")
            time.sleep(1)

def settings_menu(controller: VAPController):
    while True:
        clear_screen()
        s = controller.get_settings()
        print("===== Settings =====")
        print(f"1) Number of Tasks:        {s['number_of_tasks']}")
        print(f"2) Tasks Per Chunk:        {s['tasks_per_chunk']}")
        print(f"3) Compute Mode (1=CPU,2=GPU,3=CPU+GPU): {s['compute_mode']}")
        print(f"4) Output File:            {s['output_file']}")
        print(f"5) Task Complexity (1=simple,2=mixed,3=complex): {s['task_complexity_mode']}")
        print("----------------------------------")
        print("6) Generate/Replace Tasks Now")
        print("7) Return to Main Menu")
        choice = input("Select an option: ").strip()

        if choice == "1":
            val = input("Enter new number of tasks: ").strip()
            if val.isdigit():
                if not controller.set_number_of_tasks(int(val)):
                    print("Invalid number.")
                else:
                    print("Updated.")
            else:
                print("Invalid input.")
            time.sleep(1)
        elif choice == "2":
            val = input("Enter new tasks per chunk: ").strip()
            if val.isdigit():
                if not controller.set_tasks_per_chunk(int(val)):
                    print("Invalid number.")
                else:
                    print("Updated.")
            else:
                print("Invalid input.")
            time.sleep(1)
        elif choice == "3":
            print("\n(1) CPU\n(2) GPU\n(3) CPU+GPU")
            m_str = input("Choose compute mode (WARNING: GPU IS TEMPORARILY UNAVALIBLE): ").strip()
            if m_str.isdigit():
                if not controller.set_compute_mode(int(m_str)):
                    print("Invalid code or GPU not available.")
                else:
                    print("Compute mode updated.")
            else:
                print("Invalid input.")
            time.sleep(1)
        elif choice == "4":
            val = input("Enter output file (blank to disable): ").strip()
            controller.set_output_file(val)
            print("Output file updated.")
            time.sleep(1)
        elif choice == "5":
            print("\n(1) simple\n(2) mixed\n(3) complex")
            c_str = input("Choose task complexity mode: ").strip()
            if c_str.isdigit():
                if not controller.set_task_complexity_mode(int(c_str)):
                    print("Invalid complexity code.")
                else:
                    print("Complexity mode updated.")
            else:
                print("Invalid input.")
            time.sleep(1)
        elif choice == "6":
            controller.generate_new_tasks()
            print(f"Tasks generated. Currently {len(controller.get_tasks())} tasks in memory.")
            time.sleep(2)
        elif choice == "7":
            return
        else:
            print("Invalid choice.")
            time.sleep(1)

def manage_tasks_menu(controller: VAPController):
    while True:
        clear_screen()
        tasks_count = len(controller.get_tasks())
        print(f"===== Manage Tasks ({tasks_count} total) =====")
        print("1) Add Task")
        print("2) Remove Task")
        print("3) Modify Task")
        print("4) Show first 10 tasks")
        print("5) Clear all tasks")
        print("6) Return to Main Menu")
        choice = input("Choice: ").strip()

        if choice == "1":
            asyncio.run(add_task_interactive(controller))
        elif choice == "2":
            idx = input("Enter task index to remove: ").strip()
            if idx.isdigit():
                if controller.remove_task(int(idx)):
                    print("Removed.")
                else:
                    print("Failed. Possibly out of range.")
            else:
                print("Invalid index.")
            time.sleep(1)

        elif choice == "3":
            idx = input("Enter task index to modify: ").strip()
            if not idx.isdigit():
                print("Invalid.")
                time.sleep(1)
                continue
            i = int(idx)
            new_op = input("New op (blank=skip): ").strip()
            new_x_str = input("New x (blank=skip): ").strip()
            new_y_str = input("New y (blank=skip): ").strip()
            n_x = float(new_x_str) if new_x_str else None
            n_y = float(new_y_str) if new_y_str else None
            if controller.modify_task(i, new_op if new_op else None, n_x, n_y):
                print("Modified.")
            else:
                print("Failed or out of range.")
            time.sleep(1)

        elif choice == "4":
            clear_screen()
            tasks = controller.get_tasks()
            print("=== First 10 Tasks ===")
            for i, t in enumerate(tasks[:10], start=1):
                print(f"{i}) {t.op}({t.x}, {t.y})")
            input("\nPress Enter to continue...")
        elif choice == "5":
            controller.clear_tasks()
            print("All tasks cleared.")
            time.sleep(1)
        elif choice == "6":
            return
        else:
            print("Invalid choice.")
            time.sleep(1)

async def add_task_interactive(controller: VAPController):
    clear_screen()
    print("=== Add Task ===")
    op = input("Operation: ").strip()
    x_str = input("X value: ").strip()
    y_str = input("Y value (blank if N/A): ").strip()
    try:
        x_val = float(x_str)
        y_val = float(y_str) if y_str else None
        if not await controller.add_task(op, x_val, y_val):
            print("Failed to add task. Possibly invalid op.")
        else:
            print("Task added.")
    except ValueError:
        print("Invalid numeric input.")
    time.sleep(1)

async def do_computation(controller: VAPController):
    """
    Another emergency try/except block in case something
    big fails. We'll catch it and gracefully halt.
    """
    try:
        clear_screen()
        if not controller.get_tasks():
            print("No tasks. Generate or add tasks first.")
            time.sleep(2)
            return
        results = await controller.run_computation()
        print(f"\nDone. Computed {len(results)} results.")
        for i in range(min(5, len(results))):
            t = controller.get_tasks()[i]
            print(f"{i+1}) {t.op}({t.x},{t.y}) = {results[i]}")
        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"\nCRITICAL ERROR in do_computation: {e}")
        controller.engine.logger.exception(f"CRITICAL in do_computation: {e}")
        print("Halting gracefully. Please check logs for details.")
        time.sleep(3)
        # Return or sys.exit(1)
        return

def main():
    """
    Top-level emergency block. If something breaks here,
    we do a final fallback message.
    """
    try:
        config = ConfigManager.load()
        log_manager = LogManager(config)
        logger = log_manager.logger

        task_manager = TaskManager(config, logger)
        engine = ProcessingEngine(config, logger, task_manager)
        controller = VAPController(config, task_manager, engine)

        # Attempt checkpoint restore:
        try:
            asyncio.run(task_manager.restore())
        except Exception as e:
            logger.warning(f"Checkpoint restore error: {e}")

        # If no tasks, auto-generate:
        if not task_manager.tasks:
            try:
                init_tasks = task_manager.generate_tasks()
                task_manager.tasks = init_tasks
                logger.info(f"Loaded {len(init_tasks)} initial tasks.")
            except Exception as e:
                logger.error(f"Error generating tasks: {e}")
                print("Failed to generate tasks. Exiting gracefully.")
                return

        main_menu(controller)

    except Exception as e:
        print(f"\nFATAL ERROR in main(): {e}")
        logging.exception(f"FATAL: {e}")
        print("The script will now exit gracefully.")
        sys.exit(1)

if __name__ == "__main__":
    main()
