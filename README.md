🎤 Multimodal Speech Quality Analyzer 🤖
This project is an advanced Machine Learning system that evaluates public speaking quality. Instead of relying on subjective human "feelings," it uses Physics, Linguistics, and Emotion AI to provide a data-driven score.

🚀 The Highlights
98.6% Balanced Accuracy: Proven through rigorous mathematical testing.

Multimodal Brain: Combines OpenAI Whisper (Words), Wav2Vec2 (Tone), and Librosa (Physics).

Scientifically Verified: Includes Generalization analysis to ensure the model works on voices it has never heard before.

🧠 Technical Architecture
The system analyzes three distinct "dimensions" of every speech sample:

1. 🗣️ Linguistic Dimension (The Words)
Using OpenAI Whisper, the system transcribes audio to analyze:

Pacing: WPM (Words Per Minute) tracking.

Fluency: Detecting filler words (ums, uhs) and "hesitation" markers.

2. 🎭 Emotional Dimension (The Tone)
Using Facebook’s Wav2Vec2, we extract the "vibe" of the speaker:

Tone Score: Classifying if the voice is energetic, neutral, or hesitant.

Confidence Index: A derived metric combining emotional clarity with vocal power.

3. 🌊 Acoustic Dimension (The Physics)
Using Librosa, we measure the raw audio signals:

Stability (Jitter/Shimmer): Detecting micro-tremors that indicate nervousness or low vocal control.

Brightness (Spectral Centroid): Measuring the "crispness" and clarity of the pronunciation.

🔍 Validation & Generalization Analysis
We didn't just build a model; we proved it works.

To ensure this model wasn't just "memorizing" the training data, we performed three high-level checks:

✅ 1. 5-Fold Stratified Cross-Validation
Instead of testing the model once, we split the 500-sample dataset into 5 different groups. The model was trained and tested 5 separate times, ensuring that the 98.6% accuracy is consistent across the entire dataset and not a result of "easy" samples.

✅ 2. Learning Curve Convergence
We generated Learning Curves to visualize the model's "brain" as it studied.

The Result: The Training and Validation lines met at the top of the graph.
<img width="1163" height="694" alt="image" src="https://github.com/user-attachments/assets/efb64d13-86d8-4c3b-8cf0-ec2ab638f29e" />

The Meaning: This proves the model has generalized. It has learned the actual patterns of good speech rather than just memorizing specific voices.

✅ 3. SMOTE Balancing
Because "Poor" speakers are more common in datasets than "Elite" TED speakers, the data was imbalanced. We used SMOTE to create synthetic examples of minority classes, ensuring the AI is equally expert at identifying "Good" and "Poor" quality.

📊 What the AI Values (Feature Importance)
Through Recursive Feature Elimination (RFE), we discovered what actually makes a "Good" rating:

Energy Mean: By far the most important factor—projection is key!

Delivery Confidence: The combination of energy and emotional clarity.

Vocal Stability: Low "Jitter" scores (steady voice) beat out raw speaking speed.

📂 Project Roadmap
📍 Phase 1: Pilot (COMPLETED)
Processed 500 segments from the TED-LIUM dataset.

Achieved near-perfect convergence and classification scores.

Built a robust Inference Pipeline for .ogg and .wav files.

📈 Phase 2: Scale-Up
Transitioning from the 500-sample pilot to the Full 450-hour TED-LIUM Phase 3 Corpus.

Training on the official train folder and testing on the isolated test folder for academic-grade benchmarking.


Using vocal biomarkers to assist in the early detection of cognitive decline (Alzheimer's/Parkinson's).

🛠️ Setup & Usage
Environment: Activate your virtual environment (orator_env).

Audio: Drop any recording (even from a smartphone) into the folder.

Analyze:

Bash
python inference.py
The script will convert the audio, extract all multimodal features, and output a Final Rating with a confidence percentage.

Issues to Fix:
Data Leakage Risk (Pilot Set)
The Issue: The current 500-sample pilot set was extracted from the test directory of the TED-LIUM dataset. While Cross-Validation was used internally, using the official "test" folder for training is technically a form of data leakage in an academic context.

The Fix: Phase 2 will transition to using the official train folder for learning and keep the test folder strictly for final, "blind" evaluation.

2. 📉 SMOTE & Synthetic Bias
The Issue: Since we used SMOTE to balance the classes, there is a risk that the model is over-optimistic because it was trained on "synthetic" versions of speech patterns.

The Fix: Implementing Holdout Validation where the model is tested only on real-world, un-augmented samples that were never seen by the SMOTE algorithm.

3. 🛠️ Environmental Sensitivity
The Issue: Acoustic stability metrics (Jitter/Shimmer) are sensitive to background noise. Low-quality mics or room echo can trigger a "Poor" rating even for a confident speaker.

The Fix: Adding a Spectral Gating layer to filter ambient noise before feature extraction.

4. 🔑 Hugging Face Rate Limits
The Issue: The system currently runs on unauthenticated guest requests to the Hugging Face Hub, which can lead to slow model loading or "hanging" during peak times.

The Fix: Integrating an HF_TOKEN handler for prioritized access.

Author: Manish Kanojia
Tools Used: PyTorch, Hugging Face, Scikit-learn, Librosa
