# Nsem Tech AI: Bridging Voices, Empowering Lives 🗣️🇬🇭

![Nsem Tech Logo](https://via.placeholder.com/150x50?text=Nsem+Tech)  
*"Because Every Voice Matters"*

## 🌍 Problem Statement
In Ghana, thousands with speech impairments face daily communication barriers—unable to express basic needs in critical situations like:
- Alerting a *trotro* driver to stop ("Mate me ho")  
- Accessing healthcare or education  
- Participating in social conversations  

## 💡 Our Solution
**Nsem Tech AI** is a mobile/web app that converts text into **natural-sounding Ghanaian speech** (Twi-first, then Ewe) with:

### ✨ Key Features
| Feature | Benefit |
|---------|---------|
| **Localized TTS** | AI-trained Twi/Ewe voices (not robotic) |
| **Offline Mode** | Works without internet (downloadable language packs) |
| **Predefined Phrases** | 1-tap transport/health phrases ("Mesrɛ me wɔ ha" = "Please let me off") |
| **WhatsApp/SMS Integration** | Share speech outputs as messages |
| **Voice Customization** | Adjust pitch/speed for personalization |

## 🛠️ Technical Stack
- **TTS Engine**: Google TTS API + Mozilla TTS (fine-tuned on Twi datasets)  
- **ASR**: OpenAI Whisper (future Ghanaian accent support)  
- **Mobile**: Flutter (iOS/Android)  
- **Backend**: FastAPI (Python) + SQLite (offline cache)  

## 📲 User Flow Example  
1. **Trotro Scenario**:  
   - Open app → Tap *"Mate me ho"* → Driver hears *"I’m getting down!"*  

2. **Emergency Use**:  
   - Type *"Mepa wo kyɛw"* → App shouts *"Please help me!"* + SMS to contact  

## 🚀 Getting Started
### Prerequisites
- Flutter 3.0+  
- Python 3.8+  

### Installation
```bash
# Backend
cd backend && pip install -r requirements.txt

# Frontend
cd frontend/mobile && flutter pub get