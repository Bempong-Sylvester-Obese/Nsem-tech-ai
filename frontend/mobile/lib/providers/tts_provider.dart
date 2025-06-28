import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:audioplayers/audioplayers.dart';
import 'package:path_provider/path_provider.dart';
import '../utils/constants.dart';

class TTSProvider with ChangeNotifier {
  final AudioPlayer _audioPlayer = AudioPlayer();
  
  bool _isSynthesizing = false;
  bool _isPlaying = false;
  String _errorMessage = '';
  String? _audioFilePath;
  String _inputText = '';
  
  bool get isSynthesizing => _isSynthesizing;
  bool get isPlaying => _isPlaying;
  String get errorMessage => _errorMessage;
  String? get audioFilePath => _audioFilePath;
  String get inputText => _inputText;
  
  void setInputText(String text) {
    _inputText = text;
    notifyListeners();
  }
  
  Future<void> synthesizeText() async {
    if (_inputText.trim().isEmpty) {
      _errorMessage = 'Please enter some text to synthesize';
      notifyListeners();
      return;
    }
    
    _isSynthesizing = true;
    _errorMessage = '';
    notifyListeners();
    
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${ApiEndpoints.baseUrl}${ApiEndpoints.ttsSynthesize}'),
      );
      
      request.fields['text'] = _inputText;
      
      final response = await request.send();
      
      if (response.statusCode == 200) {
        final tempDir = await getTemporaryDirectory();
        _audioFilePath = '${tempDir.path}/synthesized_${DateTime.now().millisecondsSinceEpoch}.wav';
        
        final file = File(_audioFilePath!);
        final bytes = await response.stream.toBytes();
        await file.writeAsBytes(bytes);
        
        _errorMessage = '';
      } else {
        throw Exception('Server error: ${response.statusCode}');
      }
    } catch (e) {
      _errorMessage = 'Synthesis failed: $e';
      _audioFilePath = null;
    } finally {
      _isSynthesizing = false;
      notifyListeners();
    }
  }
  
  Future<void> playAudio() async {
    if (_audioFilePath == null) {
      _errorMessage = 'No audio file to play';
      notifyListeners();
      return;
    }
    
    try {
      _isPlaying = true;
      _errorMessage = '';
      notifyListeners();
      
      await _audioPlayer.play(DeviceFileSource(_audioFilePath!));
      
      _audioPlayer.onPlayerComplete.listen((_) {
        _isPlaying = false;
        notifyListeners();
      });
      
    } catch (e) {
      _errorMessage = 'Failed to play audio: $e';
      _isPlaying = false;
      notifyListeners();
    }
  }
  
  Future<void> stopAudio() async {
    try {
      await _audioPlayer.stop();
      _isPlaying = false;
      notifyListeners();
    } catch (e) {
      _errorMessage = 'Failed to stop audio: $e';
      notifyListeners();
    }
  }
  
  void clearError() {
    _errorMessage = '';
    notifyListeners();
  }
  
  void clearAudio() {
    _audioFilePath = null;
    _isPlaying = false;
    notifyListeners();
  }
  
  @override
  void dispose() {
    _audioPlayer.dispose();
    super.dispose();
  }
} 