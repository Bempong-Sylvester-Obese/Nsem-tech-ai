Nsem-tech-ai/
├── .github/                     # CI/CD workflows
|-- .venv                        #Virtual Environment
|-- .vscode                      .json,extensions, and config files
|
├── backend/                      # Core AI/API services
│   ├── tts_engine/               # Existing TTS (Mozilla/Whisper)
│   │   ├── train.py              # Fine-tuning scripts
│   │   └── synthesize.py         # Text-to-speech generation
│   │
│   ├── asr_engine/               # ✨ NEW: Speech-to-text
        |__preprocess.py          #Dataset Preprocessing  
│   │   ├── train.py              # ASR model training
│   │   ├── transcribe.py         # Convert speech → text
│   │   └── datasets/             # Speech samples for ASR
│   │
│   ├── api/                      # FastAPI endpoints
│   │   ├── tts.py                # Existing TTS routes
│   │   ├── asr.py                # ✨ NEW: ASR routes
│   │   └── offline_cache.py      # SQLite phrase caching
│   │
│   ├── models/                   # Pretrained models (TTS + ASR)
│   ├── cache/                    # Cached audio files
│   └── phrases.db                # SQLite phrase database
│
├── frontend/                     # User interfaces
│   ├── mobile/                   # Flutter app
│   │   ├── lib/
│   │   │   ├── tts_service.dart  # TTS API client
│   │   │   ├── asr_service.dart  # ✨ NEW: ASR API client
│   │   │   └── ...
│   │   └── ...
│   └── web/                      # (Optional) Admin dashboard
│
├── datasets/ raw_data                    # Raw data
│   ├── tts/                      # TTS training data (Twi)
│   └── asr/                      # ✨ NEW: ASR training data
│
├── scripts/                      # Utility scripts
│   ├── process_tts_data.py       # TTS dataset prep
│   ├── process_asr_data.py       # ✨ NEW: ASR dataset prep
│   └── deploy.sh                 # Deployment automation
│
├── docs/                         # Documentation
│   ├── asr_architecture.md       # ✨ NEW: ASR design
│   └── tts_architecture.md       # Existing TTS design
    |__ voicerecognition.pdf      # Voice recognition research on Eastern Languages (reference)
└── README.md                     # Updated setup guide