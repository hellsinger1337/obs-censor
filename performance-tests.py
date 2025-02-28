import time
import tracemalloc
import psutil
from audio_censor import AudioCensor  # Импортируем наш класс цензора

def test_performance(model_path, delay=1.0, sample_rate=16000, chunk_size=8000):
    print("[TEST] Starting performance test...")
    
    censor = AudioCensor(
        model_path=model_path,
        debug=False,
        overlap_seconds=0.1,
        delay_seconds=delay,
        sample_rate=sample_rate,
        chunk_size=chunk_size
    )
    
    tracemalloc.start()
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=None)
    start_mem = psutil.virtual_memory().percent
    
    try:
        censor.start()
    except KeyboardInterrupt:
        print("[TEST] Stopped manually.")
    
    end_time = time.time()
    end_cpu = psutil.cpu_percent(interval=None)
    end_mem = psutil.virtual_memory().percent
    memory_usage = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print("[TEST] Performance Results:")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Peak Memory Usage: {memory_usage[1] / 1024 / 1024:.2f} MB")
    print(f"CPU Usage Increase: {end_cpu - start_cpu:.2f}%")
    print(f"Memory Usage Increase: {end_mem - start_mem:.2f}%")
    
if __name__ == "__main__":
    test_performance("vosk-model-small-ru-0.22") 