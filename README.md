<div align="center">

# Voiceprint Identity System

<p>AI-powered voice biometric authentication and 1-to-N speaker identification platform powered by ECAPA-TDNN deep neural embeddings</p>

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/SpeechBrain-ECAPA--TDNN-8b5cf6?style=flat-square" />
<img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-8b5cf6?style=flat-square" />
<img src="https://img.shields.io/github/last-commit/tejas-coder28/Speech-Detection?style=flat-square&color=8b5cf6" />

</div>

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1428,50:8b5cf6,100:0a1428&height=48&section=header&text=🎓%20ABOUT&fontSize=22&fontColor=ffffff&fontAlignY=50" width="100%" />

Traditional authentication factors such as passwords and PINs are increasingly vulnerable to phishing, brute-force attacks, and credential theft, while physical tokens introduce user friction. This platform solves the challenge of frictionless identity verification by analyzing the unique physical geometry of human vocal tracts through deep acoustic speaker recognition models. Built with SpeechBrain's state-of-the-art ECAPA-TDNN architecture, the application extracts 192-dimensional acoustic voice embeddings from speech signals, applies spectral denoising and voice activity detection, and performs real-time 1-to-N speaker identification and zero-trust authentication through an installable Progressive Web App.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1428,50:8b5cf6,100:0a1428&height=48&section=header&text=✨%20FEATURES&fontSize=22&fontColor=ffffff&fontAlignY=50" width="100%" />

- **ECAPA-TDNN Neural Embeddings**: Computes 192-dimensional acoustic speaker embeddings via SpeechBrain's VoxCeleb-pretrained deep neural network.
- **Real-Time 1-to-N Identification**: Evaluates speech against all registered voiceprints using cosine similarity comparison and calibrated confidence thresholds.
- **Signal Denoising & Voice Activity Detection**: Preprocessing pipeline integrating Voice Activity Detection (`vad.py`) and spectral noise reduction (`denoise.py` via `noisereduce`) to strip silence and ambient room noise.
- **Multi-Tenant User Isolation**: Strict per-account data segregation ensuring registered voice profiles, audio samples, and verification logs remain fully isolated.
- **Installable Progressive Web App (PWA)**: Modern glassmorphic interface with Service Worker caching, audio visualizer waveform canvas, and mobile-friendly `MediaRecorder` support.
- **Biometric Analytics Dashboard**: Dedicated statistics console (`/stats`) tracking total enrolled speakers, verification attempts, daily accuracy curves, and confusion metrics powered by Chart.js.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1428,50:8b5cf6,100:0a1428&height=48&section=header&text=⚡%20TECH%20STACK&fontSize=22&fontColor=ffffff&fontAlignY=50" width="100%" />

| Layer | Technology |
|---|---|
| **Deep Learning & Acoustic Modeling** | PyTorch, SpeechBrain (`spkrec-ecapa-voxceleb` ECAPA-TDNN) |
| **Audio Signal Processing** | SciPy, NumPy, SoundFile, PyAV, NoiseReduce |
| **Web Backend Framework** | Python 3.10+, Flask, Gunicorn |
| **Speech Synthesis & VAD** | PyTTSx3 (Offline Text-to-Speech), Web Audio VAD |
| **Frontend Client & PWA** | HTML5, CSS3 (Glassmorphism, Neon Pulse Visualizers), JavaScript (ES6+), Chart.js 4.4, Service Workers |
| **Data Persistence & Security** | Werkzeug Security (Password Hashing), SQLite / Isolated Embedding Stores |

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1428,50:8b5cf6,100:0a1428&height=48&section=header&text=🏗️%20ARCHITECTURE&fontSize=22&fontColor=ffffff&fontAlignY=50" width="100%" />

```mermaid
flowchart TD
    subgraph Ingestion ["Audio Input Channels"]
        A1["Browser Microphone (MediaRecorder)"] --> B["app.py (Upload / Buffer)"]
        A2["Hardware Microphone (PyAudio)"] --> B
        A3["Audio File Upload (.wav, .mp3, .m4a)"] --> B
    end

    subgraph Preprocessing ["Signal Conditioning"]
        B --> C1["denoise.py (Spectral Noise Reduction)"]
        C1 --> C2["vad.py (Voice Activity Detection & Silence Trim)"]
    end

    subgraph DeepLearning ["ECAPA-TDNN Neural Network"]
        C2 --> D["model.py (SpeechBrain Pretrained Network)"]
        D --> E["Extract 192-D L2-Normalized Embedding"]
    end

    subgraph Matching ["Verification & Authentication"]
        E --> F["verify.py (Cosine Similarity vs. Stored Voiceprints)"]
        F --> G{"Similarity >= Threshold?"}
        G -- Match --> H1["Authenticated Identity (Speaker Name & Confidence Score)"]
        G -- No Match --> H2["Unknown / Impostor Rejected"]
    end

    subgraph Presentation ["Dashboard & Reporting"]
        H1 --> I["Interactive Flask PWA Client"]
        H2 --> I
        I --> J["/stats (Chart.js Accuracy Curves & Audit Logs)"]
    end
```

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1428,50:8b5cf6,100:0a1428&height=48&section=header&text=🚀%20GETTING%20STARTED&fontSize=22&fontColor=ffffff&fontAlignY=50" width="100%" />

### Prerequisites
- Python 3.10 or higher
- Git
- Microphone access (for live recording)
- FFmpeg installed on system path (recommended for multi-format audio decoding)

### 1. Clone the Repository
```bash
git clone https://github.com/tejas-coder28/Speech-Detection.git
cd Speech-Detection
```

### 2. Environment Setup & Dependencies
```bash
# Create and activate virtual environment
python -m venv venv

# Windows (Command Prompt / PowerShell)
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```
Default parameters in `.env`:
```env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
PORT=5000
REMOTE_DEPLOYMENT=0
TORCH_HOME=./pretrained_models
```

### 4. Run the Web Application
```bash
# Direct development execution
python app.py

# Or via Gunicorn (WSGI)
gunicorn app:app -b 127.0.0.1:5000
```
Open `http://127.0.0.1:5000` in your web browser to launch the voiceprint identity console.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1428,50:8b5cf6,100:0a1428&height=48&section=header&text=📁%20PROJECT%20STRUCTURE&fontSize=22&fontColor=ffffff&fontAlignY=50" width="100%" />

```
Speech-Detection/
├── docs/
│   └── assets/                   # Architecture charts & UI screenshots
├── static/
│   ├── css/style.css             # Glassmorphism styling & visualizer themes
│   ├── icon-192.png              # PWA mobile app icon
│   ├── icon-512.png              # PWA high-res app icon
│   ├── manifest.json             # Web App Manifest for mobile installation
│   └── sw.js                     # Service Worker for offline PWA caching
├── templates/
│   ├── index.html                # Main voice biometric dashboard UI
│   ├── login.html                # Multi-tenant account authentication
│   ├── register.html             # User registration interface
│   └── stats.html                # Accuracy metrics & historical verification graphs
├── tests/
│   ├── test_deletion.py          # Profile lifecycle tests
│   └── test_model.py             # Model inference & embedding tests
├── app.py                        # Flask server, route controllers & API endpoints
├── audio.py                      # Hardware microphone capture & audio verification
├── database.py                   # Multi-tenant user data store & embedding persistence
├── denoise.py                    # Spectral gating & noise reduction utilities
├── model.py                      # SpeechBrain ECAPA-TDNN model loader
├── register.py                   # Multi-segment enrollment pipeline
├── verify.py                     # Cosine similarity identification & thresholding
├── vad.py                        # Voice Activity Detection module
├── tts.py                        # Text-to-speech audio feedback generator
├── requirements.txt              # Python runtime dependencies
├── .env.example                  # Environment configuration template
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1428,50:8b5cf6,100:0a1428&height=48&section=header&text=📸%20SCREENSHOTS&fontSize=22&fontColor=ffffff&fontAlignY=50" width="100%" />

### 1. Voice Biometric Verification Interface
*Real-time speaker identification console featuring live waveform visualizer and confidence gauge.*

![Voice Verification Console](/docs/assets/01_voiceprint_verification_ui.png)
> *TODO: Capture screenshot of the main verification console at `http://127.0.0.1:5000` and save to `/docs/assets/01_voiceprint_verification_ui.png`.*

### 2. Multi-Segment Voice Enrollment
*Speaker enrollment workflow capturing three high-fidelity audio samples to synthesize the 192-D voiceprint.*

![Voice Enrollment Screen](/docs/assets/02_voice_enrollment_screen.png)
> *TODO: Capture screenshot of the enrollment modal and save to `/docs/assets/02_voice_enrollment_screen.png`.*

### 3. Biometric Analytics & Audit Statistics
*Interactive analytics dashboard tracking identification accuracy curves and per-speaker test attempts.*

![Biometric Analytics Dashboard](/docs/assets/03_biometric_analytics_stats.png)
> *TODO: Capture screenshot of `http://127.0.0.1:5000/stats` and save to `/docs/assets/03_biometric_analytics_stats.png`.*

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1428,50:8b5cf6,100:0a1428&height=48&section=header&text=🔮%20FUTURE%20IMPROVEMENTS&fontSize=22&fontColor=ffffff&fontAlignY=50" width="100%" />

- **Liveness & Anti-Spoofing Detection**: Integrate a deepfake acoustic classifier to block synthetic voice spoofing and replay attacks.
- **HNSW Vector Indexing (Qdrant / Milvus)**: Migrate from in-memory cosine matching to approximate nearest-neighbor vector search for scaling to 100,000+ enrolled identities.
- **WebAssembly (WASM) Edge Preprocessing**: Compile VAD and feature extraction routines to WebAssembly for edge filtering directly in client browsers.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1428,50:8b5cf6,100:0a1428&height=48&section=header&text=👤%20AUTHOR&fontSize=22&fontColor=ffffff&fontAlignY=50" width="100%" />

**Tejas Muralidhar**
- GitHub: [@tejas-coder28](https://github.com/tejas-coder28)
- LinkedIn: [Tejas Muralidhar](https://www.linkedin.com/in/tejasmuralidhar)
