# 🎙️ AI-Based Speaker Recognition System

An AI-powered voice recognition system built in Python that extracts 192-dimensional speaker embeddings to perform **1-to-N voice identification**. The application uses **SpeechBrain**'s pre-trained ECAPA-TDNN neural network and features a modern, dark-themed **Tkinter GUI**.

---

## 📌 Project Overview

Unlike Speech-to-Text (STT) systems that convert spoken words into text, this system identifies **who is speaking** based on their unique acoustic voiceprint:

* **Voice Registration:** Records a 15-second audio sample, extracts the speaker's embedding vector using SpeechBrain, and saves the voice profile locally.
* **Voice Identification:** Records a 5-second test snippet, computes the **Cosine Similarity** between the test audio and all registered voice profiles, and grants or denies access based on a similarity threshold.
* **Multi-Threaded Interface:** Runs long audio processing tasks on background threads to keep the desktop interface fluid and responsive.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10 / 3.11
* **Deep Learning Framework:** PyTorch & [SpeechBrain](https://speechbrain.github.io/) (`spkrec-ecapa-voxceleb`)
* **GUI Framework:** Tkinter (Custom Dark / Modern Styling)
* **Audio Processing:** `sounddevice`, `scipy`
* **Data & Math:** `numpy`, `pickle`

---

## 📂 Project Structure

```text
SPEECH DETECTION/
├── app.py              # Main Tkinter graphical interface
├── register.py         # Voice enrollment & profile creation
├── verify.py           # Speaker verification & cosine similarity logic
├── database.py         # Local storage handler (pickle dictionary)
├── audio.py            # Microphone audio recorder
├── model.py            # SpeechBrain neural network model loader
├── database/           # Directory storing speaker_dict.pkl
├── recordings/         # Folder for temporary and saved WAV files
└── requirements.txt    # Project dependencies