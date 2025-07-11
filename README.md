# Nsem Tech AI: Bridging Voices, Empowering Lives 🗣️🇬🇭

<p align="center">
  <img src="https://via.placeholder.com/150x50?text=Nsem+Tech" alt="Nsem Tech Logo" width="250"/>
</p>

<p align="center">
  <b><i>Because Every Voice Matters</i></b>
</p>

<p align="center">
  <a href="https://github.com/your-org/Nsem-tech-ai/actions"><img src="https://img.shields.io/github/workflow/status/your-org/Nsem-tech-ai/CI?label=Build&logo=github" alt="Build Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Platform-Flutter%20%7C%20Python-blueviolet" alt="Platform">
</p>

---

## 🔗 Quick Links
- [Problem Statement](#-problem-statement)
- [Solution](#-our-solution)
- [Key Features](#-key-features)
- [Technical Stack](#-technical-stack)
- [User Flow](#-user-flow-example)
- [Getting Started](#-getting-started)

---

## 🌍 Problem Statement
> In Ghana, thousands with speech impairments face daily communication barriers—unable to express basic needs in critical situations like:
>
> - 🚐 Alerting a *trotro* driver to stop ("Mate me ho")
> - 🏥 Accessing healthcare or education
> - 🗣️ Participating in social conversations

---

## 💡 Our Solution
**Nsem Tech AI** is a mobile/web app that converts text into **natural-sounding Ghanaian speech** (Twi-first, then Ewe).

> ⚡️ <b>Empowering communication, one voice at a time!</b>

---

## ✨ Key Features

| 🚀 Feature              | 🎯 Benefit                                         |
|------------------------|---------------------------------------------------|
| **Localized TTS**      | AI-trained Twi/Ewe voices (not robotic)            |
| **Offline Mode**       | Works without internet (downloadable language packs)|
| **Predefined Phrases** | 1-tap transport/health phrases                     |
| **WhatsApp/SMS**       | Share speech outputs as messages                   |
| **Voice Customization**| Adjust pitch/speed for personalization             |

---

## 🛠️ Technical Stack
- **TTS Engine**: Google TTS API + Mozilla TTS (fine-tuned on Twi datasets)
- **ASR**: OpenAI Whisper (future Ghanaian accent support)
- **Mobile**: Flutter (iOS/Android)
- **Backend**: FastAPI (Python) + SQLite (offline cache)

---

## 📲 User Flow Example

<details>
<summary><b>1. Trotro Scenario</b></summary>
<ul>
  <li>Open app → Tap <code>Mate me ho</code> → Driver hears <b>I’m getting down!</b></li>
</ul>
</details>

<details>
<summary><b>2. Emergency Use</b></summary>
<ul>
  <li>Type <code>Mepa wo kyɛw</code> → App shouts <b>Please help me!</b> + SMS to contact</li>
</ul>
</details>

---

## 🚀 Getting Started

> **Prerequisites:**
> - Flutter 3.0+
> - Python 3.8+

### 🖥️ Installation

```bash
# Backend
cd backend && pip install -r requirements.txt

# Frontend
cd frontend/mobile && flutter pub get
```

---

<p align="center">
  <sub>Made with ❤️ by the Nsem Tech Team | <a href="mailto:info@nsemtech.com">Contact Us</a></sub>
</p>
