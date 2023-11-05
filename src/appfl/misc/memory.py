import gc
import torch
import psutil
import threading

def byte2gb(x):
    """Byte to GigaByte"""
    return int(x / 2**30)

class MemoryTrace:
    """Context manager for tracking the peak memory usage of the process."""
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = byte2gb(torch.cuda.memory_allocated())
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = byte2gb(torch.cuda.memory_allocated())
        self.peak = byte2gb(torch.cuda.max_memory_allocated())
        cuda_info = torch.cuda.memory_stats()
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        self.cuda_malloc_retires = cuda_info.get("num_alloc_retries", 0)
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        self.m_cuda_ooms = cuda_info.get("num_ooms", 0)
        self.used = byte2gb(self.end - self.begin)
        self.peaked = byte2gb(self.peak - self.begin)
        self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)
