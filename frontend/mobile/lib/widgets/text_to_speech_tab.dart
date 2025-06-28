import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import '../providers/tts_provider.dart';
import '../utils/constants.dart';

class TextToSpeechTab extends StatelessWidget {
  const TextToSpeechTab({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<TTSProvider>(
      builder: (context, ttsProvider, child) {
        return Padding(
          padding: const EdgeInsets.all(AppSizes.paddingLarge),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Header
              Text(
                AppStrings.textToSpeech,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: AppColors.textPrimary,
                ),
              ),
              const SizedBox(height: AppSizes.paddingMedium),
              Text(
                'Enter Akan text to convert to speech',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: AppColors.textSecondary,
                ),
              ),
              const SizedBox(height: AppSizes.paddingXLarge),
              
              // Text Input Section
              Card(
                elevation: 2,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(AppSizes.paddingLarge),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        AppStrings.enterText,
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: AppSizes.paddingMedium),
                      TextField(
                        maxLines: 5,
                        decoration: InputDecoration(
                          hintText: 'Enter Akan text here...',
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(AppSizes.radiusSmall),
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(AppSizes.radiusSmall),
                            borderSide: const BorderSide(color: AppColors.primary),
                          ),
                        ),
                        onChanged: ttsProvider.setInputText,
                      ),
                      const SizedBox(height: AppSizes.paddingLarge),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton.icon(
                          onPressed: ttsProvider.isSynthesizing 
                              ? null 
                              : ttsProvider.synthesizeText,
                          icon: ttsProvider.isSynthesizing 
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                                  ),
                                )
                              : const Icon(Icons.synthesize),
                          label: Text(
                            ttsProvider.isSynthesizing 
                                ? AppStrings.synthesizingAudio
                                : AppStrings.synthesize,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              
              const SizedBox(height: AppSizes.paddingLarge),
              
              // Error Message
              if (ttsProvider.errorMessage.isNotEmpty)
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
                            ttsProvider.errorMessage,
                            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              color: AppColors.error,
                            ),
                          ),
                        ),
                        IconButton(
                          onPressed: ttsProvider.clearError,
                          icon: const Icon(Icons.close, color: AppColors.error),
                        ),
                      ],
                    ),
                  ),
                ),
              
              // Audio Player Section
              if (ttsProvider.audioFilePath != null)
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
                          Text(
                            'Generated Audio',
                            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: AppSizes.paddingLarge),
                          Expanded(
                            child: Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(
                                    ttsProvider.isPlaying ? Icons.pause_circle : Icons.play_circle,
                                    size: 80,
                                    color: ttsProvider.isPlaying 
                                        ? AppColors.success 
                                        : AppColors.primary,
                                  ),
                                  const SizedBox(height: AppSizes.paddingMedium),
                                  Text(
                                    ttsProvider.isPlaying 
                                        ? AppStrings.audioPlaying
                                        : AppStrings.playAudio,
                                    style: Theme.of(context).textTheme.titleMedium,
                                  ),
                                  const SizedBox(height: AppSizes.paddingLarge),
                                  Row(
                                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                                    children: [
                                      ElevatedButton.icon(
                                        onPressed: ttsProvider.isPlaying
                                            ? ttsProvider.stopAudio
                                            : ttsProvider.playAudio,
                                        icon: Icon(
                                          ttsProvider.isPlaying ? Icons.stop : Icons.play_arrow,
                                        ),
                                        label: Text(
                                          ttsProvider.isPlaying ? 'Stop' : 'Play',
                                        ),
                                      ),
                                      ElevatedButton.icon(
                                        onPressed: ttsProvider.clearAudio,
                                        icon: const Icon(Icons.clear),
                                        label: const Text('Clear'),
                                      ),
                                    ],
                                  ),
                                ],
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
} 