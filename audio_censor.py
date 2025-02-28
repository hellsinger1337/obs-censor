import pyaudio
import numpy as np
import json
import math
from vosk import Model, KaldiRecognizer
from collections import deque
from difflib import get_close_matches

class AudioCensor:
    def __init__(
        self,
        model_path,
        debug=False,
        overlap_seconds=0.1,
        delay_seconds=1.0,
        sample_rate=16000,
        chunk_size=8000
    ):
        self.model_path = model_path
        self.debug = debug
        self.overlap_seconds = overlap_seconds
        self.delay_seconds = delay_seconds
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.bad_words = set()

        if self.debug:
            print(f"[DEBUG] Initializing model from: {model_path}")

        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)

        self.beep_buffer = self.generate_beep(sample_rate=self.sample_rate)
        self.beep_length = len(self.beep_buffer)

        self.py_audio = pyaudio.PyAudio()
        self.stream_in = None
        self.stream_out = None

        self.audio_queue = deque()
        self.global_sample_count = 0

    def load_bad_words(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    self.bad_words.add(word)
        if self.debug:
            print(f"[DEBUG] Loaded bad words: {self.bad_words}")

    def is_fuzzy_bad_word(self, recognized_word):
        recognized_lower = recognized_word.lower()
        if recognized_lower in self.bad_words:
            return True
        matches = get_close_matches(recognized_lower, self.bad_words, n=1, cutoff=0.8)
        return len(matches) > 0

    def generate_beep(self, duration=0.2, frequency=1000, sample_rate=16000, volume=0.8):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        beep_signal = volume * np.sin(2 * np.pi * frequency * t)
        return beep_signal.astype(np.float32)

    def apply_beep_segment(self, float_audio, seg_start, seg_end):
        length_segment = seg_end - seg_start
        if length_segment <= 0:
            return
        if self.beep_length < length_segment:
            repeats = int(math.ceil(length_segment / self.beep_length))
            big_beep = np.tile(self.beep_buffer, repeats)[:length_segment]
            if self.debug:
                print(f"[DEBUG] Overwriting repeated beep from {seg_start} to {seg_end}")
            float_audio[seg_start:seg_end] = big_beep
        else:
            if self.debug:
                print(f"[DEBUG] Overwriting beep from {seg_start} to {seg_end}")
            target_length = min(length_segment, self.beep_length)
            float_audio[seg_start:seg_start + target_length] = self.beep_buffer[:target_length]

    def beep_in_queue(self, word_start_sample, word_end_sample):
        for (block_start, float_buf) in self.audio_queue:
            block_end = block_start + self.chunk_size
            overlap_start = max(word_start_sample, block_start)
            overlap_end = min(word_end_sample, block_end)
            if overlap_end > overlap_start:
                local_start = overlap_start - block_start
                local_end = overlap_end - block_start
                self.apply_beep_segment(float_buf, local_start, local_end)

    def start(self):
        self.stream_in = self.py_audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        self.stream_out = self.py_audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True
        )
        self.stream_in.start_stream()

        delay_chunks = int(self.delay_seconds * self.sample_rate // self.chunk_size)
        try:
            while True:
                data_block = self.stream_in.read(self.chunk_size, exception_on_overflow=False)
                float_block = np.frombuffer(data_block, dtype=np.int16).astype(np.float32) / 32767.0
                self.audio_queue.append((self.global_sample_count, float_block))

                accepted = self.recognizer.AcceptWaveform(data_block)
                if accepted:
                    final_res = json.loads(self.recognizer.Result())
                    if self.debug:
                        print("[DEBUG] Final Result:", final_res)
                    if "result" in final_res:
                        for w in final_res["result"]:
                            recognized_word = w["word"]
                            start_s = w["start"] - self.overlap_seconds
                            end_s = w["end"] + self.overlap_seconds
                            if start_s < 0:
                                start_s = 0.0
                            word_start_sample = int(start_s * self.sample_rate)
                            word_end_sample = int(end_s * self.sample_rate)
                            if self.is_fuzzy_bad_word(recognized_word):
                                if self.debug:
                                    print(f"[DEBUG] Found bad word: '{recognized_word}' with overlap")
                                self.beep_in_queue(word_start_sample, word_end_sample)
                else:
                    partial_res = json.loads(self.recognizer.PartialResult())
                    if self.debug:
                        print("[DEBUG] Partial Result:", partial_res)

                if len(self.audio_queue) > delay_chunks:
                    out_start, out_float_buf = self.audio_queue.popleft()
                    out_data = (out_float_buf * 32767.0).astype(np.int16).tobytes()
                    self.stream_out.write(out_data)

                self.global_sample_count += self.chunk_size
        except KeyboardInterrupt:
            if self.debug:
                print("[DEBUG] Stopped by user")
        finally:
            self._drain_queue()
            self.stop()

    def _drain_queue(self):
        while self.audio_queue:
            _, out_float_buf = self.audio_queue.popleft()
            out_data = (out_float_buf * 32767.0).astype(np.int16).tobytes()
            self.stream_out.write(out_data)

    def stop(self):
        if self.stream_in is not None:
            self.stream_in.stop_stream()
            self.stream_in.close()
        if self.stream_out is not None:
            self.stream_out.stop_stream()
            self.stream_out.close()
        self.py_audio.terminate()


def main():
    censor = AudioCensor(
        model_path="vosk-model-small-ru-0.22",
        debug=True,
        overlap_seconds=0.1,
        delay_seconds=2.0,
        sample_rate=16000,
        chunk_size=8000
    )
    censor.bad_words = {"блять", "сука", "нахуй", "один"}
    censor.start()

if __name__ == "__main__":
    main()