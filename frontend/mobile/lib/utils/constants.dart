import 'package:flutter/material.dart';

class AppColors {
  static const Color primary = Color(0xFF2563EB);
  static const Color secondary = Color(0xFF3B82F6);
  static const Color accent = Color(0xFF60A5FA);
  static const Color background = Color(0xFFF8FAFC);
  static const Color surface = Color(0xFFFFFFFF);
  static const Color textPrimary = Color(0xFF1E293B);
  static const Color textSecondary = Color(0xFF64748B);
  static const Color success = Color(0xFF10B981);
  static const Color error = Color(0xFFEF4444);
  static const Color warning = Color(0xFFF59E0B);
}

class ApiEndpoints {
  static const String baseUrl = 'http://localhost:8000';
  static const String asrTranscribe = '/transcribe';
  static const String asrTrain = '/train';
  static const String ttsSynthesize = '/synthesize';
  static const String ttsTrain = '/train';
}

class AppStrings {
  static const String appName = 'Nsem Tech AI';
  static const String appDescription = 'Akan Speech Recognition & TTS';
  
  // Home Screen
  static const String speechToText = 'Speech to Text';
  static const String textToSpeech = 'Text to Speech';
  static const String recordAudio = 'Record Audio';
  static const String uploadAudio = 'Upload Audio';
  static const String enterText = 'Enter Akan Text';
  static const String synthesize = 'Synthesize';
  static const String playAudio = 'Play Audio';
  
  // Messages
  static const String recordingStarted = 'Recording started...';
  static const String recordingStopped = 'Recording stopped';
  static const String processingAudio = 'Processing audio...';
  static const String synthesizingAudio = 'Synthesizing audio...';
  static const String audioPlaying = 'Playing audio...';
  static const String audioStopped = 'Audio stopped';
  static const String noAudioFile = 'No audio file selected';
  static const String transcriptionSuccess = 'Transcription completed';
  static const String synthesisSuccess = 'Audio synthesis completed';
  
  // Errors
  static const String permissionDenied = 'Microphone permission denied';
  static const String networkError = 'Network error occurred';
  static const String audioError = 'Audio processing error';
  static const String fileError = 'File upload error';
}

class AppSizes {
  static const double paddingSmall = 8.0;
  static const double paddingMedium = 16.0;
  static const double paddingLarge = 24.0;
  static const double paddingXLarge = 32.0;
  
  static const double radiusSmall = 8.0;
  static const double radiusMedium = 12.0;
  static const double radiusLarge = 16.0;
  
  static const double iconSize = 24.0;
  static const double buttonHeight = 48.0;
} 