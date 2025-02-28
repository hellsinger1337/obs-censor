import time
import tracemalloc
import psutil
import matplotlib.pyplot as plt
import threading
from audio_censor import AudioCensor

def monitor_performance(cpu_usage, mem_usage, stop_event):
    while not stop_event.is_set():
        cpu_usage.append(psutil.cpu_percent(interval=0.5))
        mem_usage.append(psutil.virtual_memory().percent)

def test_performance(model_path, delay=2.0, sample_rate=16000, chunk_size=8000, bad_words_file="badwords.txt"):
    print("[TEST] Starting performance test...")
    
    censor = AudioCensor(
        model_path=model_path,
        debug=False,
        overlap_seconds=0.1,
        delay_seconds=delay,
        sample_rate=sample_rate,
        chunk_size=chunk_size
    )
    
    censor.load_bad_words(bad_words_file)
    
    tracemalloc.start()
    start_time = time.time()
    
    cpu_usage = []
    mem_usage = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_performance, args=(cpu_usage, mem_usage, stop_event))
    monitor_thread.start()
    
    try:
        censor.start()
    except KeyboardInterrupt:
        print("[TEST] Stopped manually.")
    
    stop_event.set()
    monitor_thread.join()
    
    end_time = time.time()
    memory_usage = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print("[TEST] Performance Results:")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Peak Memory Usage: {memory_usage[1] / 1024 / 1024:.2f} MB")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cpu_usage, label='CPU Usage (%)')
    plt.xlabel("Time (0.5s intervals)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(mem_usage, label='Memory Usage (%)', color='red')
    plt.xlabel("Time (0.5s intervals)")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage Over Time")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    test_performance("vosk-model-small-ru-0.22", bad_words_file="badwords.txt")
