import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import '../providers/audio_provider.dart';
import '../utils/constants.dart';

class SpeechToTextTab extends StatelessWidget {
  const SpeechToTextTab({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<AudioProvider>(
      builder: (context, audioProvider, child) {
        return Padding(
          padding: const EdgeInsets.all(AppSizes.paddingLarge),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Header
              Text(
                AppStrings.speechToText,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: AppColors.textPrimary,
                ),
              ),
              const SizedBox(height: AppSizes.paddingMedium),
              Text(
                'Record or upload audio to convert Akan speech to text',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: AppColors.textSecondary,
                ),
              ),
              const SizedBox(height: AppSizes.paddingXLarge),
              
              // Recording Section
              Card(
                elevation: 2,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(AppSizes.paddingLarge),
                  child: Column(
                    children: [
                      Icon(
                        audioProvider.isRecording ? Icons.stop_circle : Icons.mic,
                        size: 64,
                        color: audioProvider.isRecording 
                            ? AppColors.error 
                            : AppColors.primary,
                      ),
                      const SizedBox(height: AppSizes.paddingMedium),
                      Text(
                        audioProvider.isRecording 
                            ? AppStrings.recordingStarted
                            : AppStrings.recordAudio,
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: AppSizes.paddingLarge),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          ElevatedButton.icon(
                            onPressed: audioProvider.isRecording
                                ? audioProvider.stopRecording
                                : audioProvider.startRecording,
                            icon: Icon(
                              audioProvider.isRecording ? Icons.stop : Icons.mic,
                            ),
                            label: Text(
                              audioProvider.isRecording ? 'Stop' : 'Record',
                            ),
                          ),
                          ElevatedButton.icon(
                            onPressed: () => _pickAudioFile(context, audioProvider),
                            icon: const Icon(Icons.upload_file),
                            label: const Text(AppStrings.uploadAudio),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
              
              const SizedBox(height: AppSizes.paddingLarge),
              
              // Processing Indicator
              if (audioProvider.isProcessing)
                Card(
                  elevation: 2,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(AppSizes.paddingLarge),
                    child: Row(
                      children: [
                        const SpinKitWave(
                          color: AppColors.primary,
                          size: 24,
                        ),
                        const SizedBox(width: AppSizes.paddingMedium),
                        Text(
                          AppStrings.processingAudio,
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                      ],
                    ),
                  ),
                ),
              
              // Error Message
              if (audioProvider.errorMessage.isNotEmpty)
                Card(
                  color: AppColors.error.withOpacity(0.1),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(AppSizes.paddingMedium),
                    child: Row(
                      children: [
                        const Icon(
                          Icons.error_outline,
                          color: AppColors.error,
                        ),
                        const SizedBox(width: AppSizes.paddingSmall),
                        Expanded(
                          child: Text(
                            audioProvider.errorMessage,
                            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              color: AppColors.error,
                            ),
                          ),
                        ),
                        IconButton(
                          onPressed: audioProvider.clearError,
                          icon: const Icon(Icons.close, color: AppColors.error),
                        ),
                      ],
                    ),
                  ),
                ),
              
              // Transcription Result
              if (audioProvider.transcription.isNotEmpty)
                Expanded(
                  child: Card(
                    elevation: 2,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(AppSizes.paddingLarge),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                'Transcription',
                                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              IconButton(
                                onPressed: audioProvider.clearTranscription,
                                icon: const Icon(Icons.clear),
                              ),
                            ],
                          ),
                          const SizedBox(height: AppSizes.paddingMedium),
                          Expanded(
                            child: Container(
                              width: double.infinity,
                              padding: const EdgeInsets.all(AppSizes.paddingMedium),
                              decoration: BoxDecoration(
                                color: AppColors.background,
                                borderRadius: BorderRadius.circular(AppSizes.radiusSmall),
                                border: Border.all(color: AppColors.textSecondary.withOpacity(0.2)),
                              ),
                              child: SingleChildScrollView(
                                child: Text(
                                  audioProvider.transcription,
                                  style: Theme.of(context).textTheme.bodyMedium,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }
  
  Future<void> _pickAudioFile(BuildContext context, AudioProvider audioProvider) async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.audio,
        allowedExtensions: ['wav', 'mp3'],
      );
      
      if (result != null && result.files.single.path != null) {
        await audioProvider.uploadAudioFile(result.files.single.path!);
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to pick audio file: $e'),
            backgroundColor: AppColors.error,
          ),
        );
      }
    }
  }
} 