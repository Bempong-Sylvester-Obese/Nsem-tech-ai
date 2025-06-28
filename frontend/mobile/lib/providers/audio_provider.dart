import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import '../utils/constants.dart';

class AudioProvider with ChangeNotifier {
  final Record _audioRecorder = Record();
  
  bool _isRecording = false;
  bool _isProcessing = false;
  String _transcription = '';
  String _errorMessage = '';
  String? _audioFilePath;
  
  bool get isRecording => _isRecording;
  bool get isProcessing => _isProcessing;
  String get transcription => _transcription;
  String get errorMessage => _errorMessage;
  String? get audioFilePath => _audioFilePath;
  
  Future<bool> requestPermissions() async {
    final status = await Permission.microphone.request();
    return status.isGranted;
  }
  
  Future<void> startRecording() async {
    try {
      final hasPermission = await requestPermissions();
      if (!hasPermission) {
        _errorMessage = AppStrings.permissionDenied;
        notifyListeners();
        return;
      }
      
      final tempDir = await getTemporaryDirectory();
      _audioFilePath = '${tempDir.path}/recording_${DateTime.now().millisecondsSinceEpoch}.wav';
      
      await _audioRecorder.start(
        path: _audioFilePath!,
        encoder: AudioEncoder.wav,
        bitRate: 16000,
        samplingRate: 16000,
      );
      
      _isRecording = true;
      _errorMessage = '';
      notifyListeners();
    } catch (e) {
      _errorMessage = 'Failed to start recording: $e';
      notifyListeners();
    }
  }
  
  Future<void> stopRecording() async {
    try {
      await _audioRecorder.stop();
      _isRecording = false;
      notifyListeners();
      
      if (_audioFilePath != null) {
        await transcribeAudio();
      }
    } catch (e) {
      _errorMessage = 'Failed to stop recording: $e';
      _isRecording = false;
      notifyListeners();
    }
  }
  
  Future<void> transcribeAudio() async {
    if (_audioFilePath == null) {
      _errorMessage = AppStrings.noAudioFile;
      notifyListeners();
      return;
    }
    
    _isProcessing = true;
    _errorMessage = '';
    notifyListeners();
    
    try {
      final file = File(_audioFilePath!);
      if (!await file.exists()) {
        throw Exception('Audio file not found');
      }
      
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${ApiEndpoints.baseUrl}${ApiEndpoints.asrTranscribe}'),
      );
      
      request.files.add(
        await http.MultipartFile.fromPath('audio', _audioFilePath!),
      );
      
      final response = await request.send();
      final responseBody = await response.stream.bytesToString();
      
      if (response.statusCode == 200) {
        // Parse JSON response
        final data = responseBody; // You might want to use jsonDecode here
        _transcription = data.contains('"text"') 
            ? data.split('"text"')[1].split('"')[1] 
            : data;
        _errorMessage = '';
      } else {
        throw Exception('Server error: ${response.statusCode}');
      }
    } catch (e) {
      _errorMessage = 'Transcription failed: $e';
      _transcription = '';
    } finally {
      _isProcessing = false;
      notifyListeners();
    }
  }
  
  Future<void> uploadAudioFile(String filePath) async {
    _audioFilePath = filePath;
    await transcribeAudio();
  }
  
  void clearTranscription() {
    _transcription = '';
    _errorMessage = '';
    notifyListeners();
  }
  
  void clearError() {
    _errorMessage = '';
    notifyListeners();
  }
  
  @override
  void dispose() {
    _audioRecorder.dispose();
    super.dispose();
  }
} 