# 🎙️ Voiceprint Identity System

An AI-powered voice biometric authentication and speaker identification web application. Built with **Flask**, **SpeechBrain (ECAPA-TDNN)**, **PyTorch**, and modern **PWA (Progressive Web App)** frontend technologies, the system extracts 192-dimensional acoustic speaker embeddings to perform 1-to-N voice identification with high precision.

---

## 📌 Features

### 🔐 Voice Biometrics & Authentication
- **Multi-Tenant User Isolation**: Strict per-account data isolation (`database.safe_user_id()`). Each account maintains independent database profiles, recordings folders, and verification logs with zero cross-account data leakage.
- **Voice Enrollment**: Enrolls speakers using three 5-second audio segments (15 seconds total) to build a robust 192-D acoustic voiceprint.
- **1-to-N Speaker Verification**: Identifies unknown speakers against all enrolled voice profiles using Cosine Similarity matching against tuned confidence thresholds.
- **Audio Integrity Checks**: Integrated Voice Activity Detection (VAD), silence filtering, and multi-speaker detection to prevent noisy or corrupted profile enrollments.
- **Dual Recording Paths**:
  - **Desktop Hardware Recording**: Direct server-side PyAudio capture for local desktop deployments.
  - **Mobile Browser Recording**: HTML5 `MediaRecorder` API capture with stream buffering for smartphones and mobile PWA viewports.
  - **File Upload Support**: Accepts uploaded audio clips (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`, `.mp4`).

### 📱 Progressive Web App (PWA) & UI Design
- **Installable Mobile PWA**: Native app-like experience with Web App Manifest, Service Worker shell caching, standalone display mode, and custom app icons.
- **Modern Glassmorphism UI**: Obsidian gradient mesh styling with glowing neon accents, live visualizer pulse waves, circular confidence gauges, and real-time audio waveform previews.
- **Responsive Layout**: Touch-friendly interface optimized for screen widths ranging from mobile phones to desktop displays.

### 📊 Analytics & Diagnostic Tools
- **Roster & Accuracy Analytics**: Interactive dashboard (`/dashboard/stats`) tracking total enrolled speakers, verification runs, success/failure counts, overall accuracy, daily accuracy trends, and per-speaker test attempts using **Chart.js**.
- **Diagnostic Inspection**: Displays confidence scores, runner-up identities, and margin thresholds for testing and tuning.
- **Testing Mode**: Access via `/?test=1` to attach ground-truth speaker labels during evaluation runs.

### 🌐 Cloud & Remote Deployment Ready
- **Client-Side TTS Playback**: Reworked Text-to-Speech (pyttsx3 `save_to_file`) engine that renders audio greetings to `/static/tts/` for HTML `<audio>` playback on headless remote servers.
- **Remote Flag (`REMOTE_DEPLOYMENT`)**: Environment variable to gracefully hide server hardware microphone controls when deployed to cloud hosts (Heroku, Render, AWS, Railway), defaulting to audio file uploads and mobile browser recording.
- **Production WSGI**: Configured with **Gunicorn** for production server execution.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.11, Flask, Gunicorn, PyTorch, [SpeechBrain](https://speechbrain.github.io/) (`spkrec-ecapa-voxceleb`), PyAV, SoundFile, SciPy, NumPy, NoiseReduce, PyTTSx3
- **Frontend**: Vanilla HTML5, Custom CSS3 Design System (Vanilla CSS with CSS Variables & Glassmorphism), JavaScript (ES6+), Chart.js 4.4, MediaRecorder API, Service Worker
- **Storage & Security**: Password hashing (`werkzeug.security`), isolated `.pkl` embedding databases, CSV logging

---

## 📂 Project Structure

```text
Speech Detection/
├── app.py                  # Main Flask web application & REST API endpoints
├── model.py                # SpeechBrain ECAPA-TDNN neural network loader
├── verify.py               # Cosine similarity identification & threshold logic
├── register.py             # Multi-segment voice enrollment pipeline
├── database.py             # Per-user isolated speaker storage handler
├── audio.py                # Hardware microphone recording & signal checks
├── tts.py                  # Text-to-Speech greeting file generator
├── vad.py                  # Voice Activity Detection (VAD) module
├── denoise.py              # Audio noise reduction & filtering
├── analyze_test_log.py     # Verification log parsing & statistical analytics
├── requirements.txt        # Python package dependencies
├── .gitignore              # Excludes sensitive user data, audio clips, and databases
├── static/
│   ├── css/style.css       # Responsive CSS stylesheet & design tokens
│   ├── manifest.json       # PWA web manifest
│   ├── sw.js               # Service worker shell caching script
│   ├── icon-192.png        # PWA app icon (192x192)
│   ├── icon-512.png        # PWA app icon (512x512)
│   └── tts/                # Generated audio greeting cache
└── templates/
    ├── index.html          # Main speaker recognition dashboard
    ├── login.html          # Authentication login screen
    ├── register.html       # User signup screen
    └── stats.html          # Analytics dashboard
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Virtualenv (`python -m venv venv311`)

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/voiceprint-identity-system.git
   cd voiceprint-identity-system
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv311
   # On Windows:
   .\venv311\Scripts\activate
   # On macOS/Linux:
   source venv311/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python app.py
   ```

5. **Access the Web App**:
   Open `http://localhost:5000` in your web browser. Create an account to log in and start enrolling voice profiles.

---

## ☁️ Remote Cloud Deployment

When deploying to cloud platforms (such as Render, Heroku, Railway, or AWS):

1. **Set Environment Variables**:
   ```bash
   SECRET_KEY="your-strong-production-secret-key"
   REMOTE_DEPLOYMENT="true"
   ```

2. **Start with Gunicorn**:
   ```bash
   gunicorn --bind 0.0.0.0:5000 app:app
   ```

---

## 🔒 Privacy & Security

- **Data Isolation**: User data, voice profiles, and verification logs are kept strictly separated by account ID.
- **Git Exclusions**: All sensitive user directories (`database/`, `recordings/`, `test_logs/`, `users.json`), raw audio files (`*.wav`), and model checkpoints are excluded via `.gitignore` to prevent committing private biometric data to source control.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).