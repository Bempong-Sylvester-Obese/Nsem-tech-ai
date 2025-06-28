# Nsem Tech AI - Flutter Mobile App

A Flutter mobile application for Akan Speech Recognition and Text-to-Speech functionality.

## Features

- **Speech-to-Text**: Record or upload audio files to convert Akan speech to text
- **Text-to-Speech**: Enter Akan text to synthesize natural speech
- **Real-time Recording**: Built-in audio recording with permission handling
- **File Upload**: Support for uploading existing audio files (WAV, MP3)
- **Audio Playback**: Built-in audio player for synthesized speech
- **Modern UI**: Clean, intuitive interface with Material Design

## Prerequisites

- Flutter SDK (>=3.0.0)
- Dart SDK (>=3.0.0)
- Android Studio / VS Code
- iOS Simulator (for iOS development)
- Android Emulator (for Android development)

## Setup Instructions

### 1. Install Flutter Dependencies

```bash
cd frontend/mobile
flutter pub get
```

### 2. Configure Backend URL

Update the backend URL in `lib/utils/constants.dart`:

```dart
class ApiEndpoints {
  static const String baseUrl = 'http://YOUR_BACKEND_IP:8000';
  // ... other endpoints
}
```

### 3. Platform-Specific Setup

#### Android
- Ensure Android SDK is installed
- Create/update Android Virtual Device (AVD) if needed
- Permissions are already configured in `android/app/src/main/AndroidManifest.xml`

#### iOS
- Ensure Xcode is installed (macOS only)
- Permissions are already configured in `ios/Runner/Info.plist`

### 4. Run the App

```bash
# For Android
flutter run -d android

# For iOS
flutter run -d ios

# For web (optional)
flutter run -d chrome
```

## Project Structure

```
lib/
├── main.dart                 # App entry point
├── providers/                # State management
│   ├── audio_provider.dart   # Speech recognition logic
│   └── tts_provider.dart     # Text-to-speech logic
├── screens/                  # Main screens
│   └── home_screen.dart      # Main app screen with tabs
├── widgets/                  # Reusable UI components
│   ├── speech_to_text_tab.dart
│   └── text_to_speech_tab.dart
└── utils/                    # Utilities and constants
    └── constants.dart        # App constants and configuration
```

## Usage

### Speech-to-Text
1. Tap the "Speech to Text" tab
2. Choose to either:
   - **Record Audio**: Tap the microphone button to start recording
   - **Upload Audio**: Tap "Upload Audio" to select an existing file
3. Wait for processing to complete
4. View the transcription result

### Text-to-Speech
1. Tap the "Text to Speech" tab
2. Enter Akan text in the text field
3. Tap "Synthesize" to generate audio
4. Use the audio player to play/stop the synthesized speech

## Dependencies

### Core Dependencies
- `flutter`: Flutter framework
- `provider`: State management
- `http`: HTTP requests for API calls
- `dio`: Advanced HTTP client

### Audio Dependencies
- `audioplayers`: Audio playback
- `record`: Audio recording
- `permission_handler`: Permission management

### UI Dependencies
- `google_fonts`: Custom fonts
- `flutter_spinkit`: Loading animations
- `file_picker`: File selection

## API Integration

The app communicates with the backend API endpoints:

- `POST /transcribe` - Speech recognition
- `POST /synthesize` - Text-to-speech synthesis
- `POST /train` - Model training (for future use)

## Troubleshooting

### Common Issues

1. **Permission Denied**
   - Ensure microphone permissions are granted
   - Check platform-specific permission settings

2. **Network Error**
   - Verify backend server is running
   - Check API endpoint URL in constants
   - Ensure network connectivity

3. **Audio Recording Issues**
   - Check microphone permissions
   - Verify audio file format support
   - Ensure sufficient storage space

### Debug Mode

Run with debug flags for detailed logging:

```bash
flutter run --debug
```

## Building for Production

### Android APK
```bash
flutter build apk --release
```

### iOS IPA
```bash
flutter build ios --release
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the Nsem Tech AI project. See the main project LICENSE file for details. 