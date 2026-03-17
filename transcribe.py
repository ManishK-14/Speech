import os
import re
import json
import warnings
from collections import Counter
import torch
import numpy as np
import librosa
import whisper
import matplotlib.pyplot as plt
from scipy.signal import medfilt

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

warnings.filterwarnings("ignore")

class AIPublicSpeakingCoach:
    def __init__(self, model_size="small"):
        print("=== Initializing AI Public Speaking Coach ===")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper '{model_size}' model on {self.device.upper()}...")
        self.model = whisper.load_model(model_size, device=self.device)
        
        self.target_sr = 16000
        self.fillers = ["um", "uh", "ah", "er", "hmm", "like", "so", "basically", "actually"]
        self.ideal_wpm_range = (130, 150)
        self.min_pause_sec = 0.5
        
    def analyze_audio(self, file_path):

        print(f"\nProcessing File: {file_path}")
        
        if not os.path.exists(file_path):
            print("Error: File not found!")
            return None

        print("Loading audio and extracting base features:")
        y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        
        if HAS_NOISEREDUCE:
            print("Applying noise reduction...")
            y_clean = nr.reduce_noise(y=y_trimmed, sr=sr, prop_decrease=0.8)
        else:
            y_clean = y_trimmed

        print("Running Whisper AI transcription...")
        transcription_result = self.model.transcribe(file_path, word_timestamps=True)
        transcript = transcription_result.get("text", "").strip()
        words = transcript.split()
        word_count = len(words)
        
        print("Analyzing verbal content (fillers, repetitions)...")
        filler_counts = {}
        total_fillers = 0
        
        # Parse Whisper word-level timestamps to find fillers
        for segment in transcription_result.get("segments", []):
            for word_info in segment.get("words", []):
                clean_word = word_info.get("word", "").strip().lower().strip(".,!?;:")
                if clean_word in self.fillers:
                    filler_counts[clean_word] = filler_counts.get(clean_word, 0) + 1
                    total_fillers += 1

        # Calculate Pace (Words Per Minute)
        wpm = (word_count / duration) * 60 if duration > 0 else 0

        # --- STEP 4: Prosody & Acoustic Analysis (The ML part) ---
        print("Extracting acoustic features (Pitch, Energy, Voice Quality)...")
        
        # 4a. Energy (Volume/RMS)
        rms = librosa.feature.rms(y=y_clean)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
        avg_energy = float(np.mean(rms_db))
        
        # 4b. Pauses (Silence detection)
        intervals = librosa.effects.split(y_clean, top_db=25)
        pauses = []
        for i in range(1, len(intervals)):
            gap_start = intervals[i - 1][1] / sr
            gap_end = intervals[i][0] / sr
            pause_duration = gap_end - gap_start
            if pause_duration >= self.min_pause_sec:
                pauses.append({
                    "start": float(round(gap_start, 2)), 
                    "duration": float(round(pause_duration, 2))
                })
        
        total_pause_time = sum(p["duration"] for p in pauses)

        f0, _, _ = librosa.pyin(y_clean, fmin=65, fmax=2000, sr=sr, frame_length=2048)
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) > 10:
            f0_smooth = medfilt(f0_voiced, kernel_size=5)
            mean_pitch = float(np.mean(f0_smooth))
            pitch_std = float(np.std(f0_smooth))
            pitch_cv = (pitch_std / mean_pitch) * 100 if mean_pitch > 0 else 0
        else:
            mean_pitch, pitch_cv = 0, 0
            f0_smooth = []

        # 4d. Voice Quality Metrics (Jitter & Shimmer)
        jitter, shimmer = 0.0, 0.0
        if len(f0_voiced) > 30:
            # Jitter: frequency perturbation
            periods = 1.0 / f0_voiced
            period_diffs = np.abs(np.diff(periods))
            jitter = (np.mean(period_diffs) / np.mean(periods)) * 100
            
            # Shimmer: amplitude perturbation
            hop = 2048 // 4
            rms_frames = librosa.feature.rms(y=y_clean, frame_length=2048, hop_length=hop)[0]
            if len(rms_frames) > 1:
                amp_diffs = np.abs(np.diff(rms_frames))
                shimmer = (np.mean(amp_diffs) / (np.mean(rms_frames) + 1e-10)) * 100

        # --- STEP 5: Generate Coaching Feedback ---
        print("Generating feedback...")
        feedback = self.generate_feedback(wpm, total_fillers, pitch_cv, jitter)

        # --- STEP 6: Compile Results ---
        # ALL numbers are explicitly cast to int() or float() to prevent JSON crashes
        report = {
            "metadata": {
                "file": file_path,
                "duration_sec": float(round(duration, 2)),
                "word_count": int(word_count)
            },
            "transcription": transcript,
            "metrics": {
                "wpm": float(round(wpm, 1)),
                "total_fillers": int(total_fillers),
                "filler_breakdown": {str(k): int(v) for k, v in filler_counts.items()},
                "total_pauses": int(len(pauses)),
                "pause_time_sec": float(round(total_pause_time, 2)),
                "mean_pitch_hz": float(round(mean_pitch, 1)),
                "pitch_variation_cv": float(round(pitch_cv, 1)),
                "vocal_jitter_pct": float(round(jitter, 2)),
                "vocal_shimmer_pct": float(round(shimmer, 2))
            },
            "feedback": feedback
        }

        self.plot_features(y_clean, sr, rms_db, f0_smooth, file_path)

        return report

    def generate_feedback(self, wpm, fillers, pitch_cv, jitter):
        """Rules-based engine to provide actionable public speaking advice."""
        tips = []
        
        # Pace
        if wpm < self.ideal_wpm_range[0]:
            tips.append(f"Pace: Slow ({round(wpm)} WPM). Try to speak a bit faster to keep the audience engaged.")
        elif wpm > self.ideal_wpm_range[1]:
            tips.append(f"Pace: Fast ({round(wpm)} WPM). Slow down slightly to ensure clarity.")
        else:
            tips.append(f"Pace: Excellent ({round(wpm)} WPM). You are speaking at a very professional rate.")

        # Fillers
        if fillers > 5:
            tips.append(f"Fillers: High ({fillers} detected). Try to replace words like 'um' and 'like' with silent pauses.")
        elif fillers > 0:
            tips.append(f"Fillers: Controlled ({fillers} detected). Good job keeping filler words to a minimum.")
            
        if pitch_cv < 12.0:
            tips.append("Tone: Monotone. Try to add more vocal variety and enthusiasm to highlight key points.")
        else:
            tips.append("Tone: Expressive. You have good vocal variation.")

        if jitter > 2.0:
            tips.append("Voice Quality: Elevated jitter detected. Take deep breaths; this often indicates vocal tension or nervousness.")

        return tips

    def plot_features(self, y, sr, rms_db, f0, filepath):
        plt.figure(figsize=(12, 8))
        
        # Plot Waveform
        plt.subplot(3, 1, 1)
        times = np.arange(len(y)) / sr
        plt.plot(times, y, color="steelblue", alpha=0.7)
        plt.title("Audio Waveform")
        plt.ylabel("Amplitude")

        plt.subplot(3, 1, 2)
        if len(f0) > 0:
            plt.plot(f0, color="coral", marker=".", markersize=2, linestyle="None")
            plt.title("Pitch Contour (Fundamental Frequency)")
            plt.ylabel("Frequency (Hz)")

        # Plot Energy
        plt.subplot(3, 1, 3)
        rms_times = librosa.times_like(rms_db, sr=sr)
        plt.plot(rms_times, rms_db, color="forestgreen")
        plt.title("Vocal Energy (RMS dB)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("dB")

        plt.tight_layout()
        filename = "analysis_plot.png"
        plt.savefig(filename, dpi=150)
        print(f"Visualization saved to {filename}")
        plt.close()


if __name__ == "__main__":

    coach = AIPublicSpeakingCoach(model_size="base")

    audio_file = "Sample audio.ogg" 
    report = coach.analyze_audio(audio_file)

    if report:
        print("\n" + "="*50)
        print("          SPEECH ANALYSIS REPORT")
        print("="*50)
        
        print("\n[ METRICS ]")
        for key, value in report["metrics"].items():
            if key == "filler_breakdown": continue
            print(f" - {key.replace('_', ' ').title()}: {value}")
            
        if report["metrics"]["total_fillers"] > 0:
            print(f" - Filler Breakdown: {report['metrics']['filler_breakdown']}")

        print("\n[ COACHING FEEDBACK ]")
        for tip in report["feedback"]:
            print(f" 💡 {tip}")

        # Saving report to JSON
        with open("report.json", "w") as f:
            json.dump(report, f, indent=4)
        print("\n[ Full report saved to report.json ]")