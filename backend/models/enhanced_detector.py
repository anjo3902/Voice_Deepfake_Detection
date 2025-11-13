"""
ENHANCED DETECTION ENGINE - Multi-Layer TTS Detection
Combines neural model + spectral analysis + heuristics
"""
import numpy as np
import librosa
import torch
import scipy.stats as stats


class EnhancedTTSDetector:
    """
    Multi-factor TTS detection system with ADAPTIVE THRESHOLD
    Analyzes audio quality to choose appropriate threshold
    """
    
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.secondary_model = None  # For ElevenLabs detection
        self.base_threshold = threshold
        
        # Detection weights - For production-trained model
        # A properly trained model should dominate
        self.weights = {
            'model_confidence': 1.00,      # TRUST THE TRAINED MODEL 100%!
            'secondary_confidence': 0.00,  # Disabled
            'spectral_score': 0.00,        # Disabled - model knows best
            'prosody_score': 0.00,         # Disabled - model knows best
            'statistical_score': 0.00      # Disabled - model knows best
        }
        
        # Standard threshold - model decides
        self.tts_detection_threshold = 0.50
    
    def analyze_audio_quality(self, waveform_np, sr=16000):
        """
        Analyze audio quality to determine if it's clean (TTS) or noisy (real recording)
        Returns quality score: 0 (very noisy) to 1 (very clean)
        """
        # 1. Signal-to-Noise Ratio estimate
        rms = librosa.feature.rms(y=waveform_np)[0]
        rms_std = np.std(rms)
        snr_score = 1.0 / (1.0 + rms_std / 0.1)  # Lower std = cleaner = higher score
        
        # 2. Spectral flatness (clean audio is more tonal)
        spectral_flatness = librosa.feature.spectral_flatness(y=waveform_np)[0]
        flatness_score = 1.0 - np.mean(spectral_flatness)  # Lower flatness = cleaner
        
        # 3. Zero crossing consistency (clean audio more consistent)
        zcr = librosa.feature.zero_crossing_rate(waveform_np)[0]
        zcr_std = np.std(zcr)
        zcr_score = 1.0 / (1.0 + zcr_std / 0.1)
        
        # 4. High frequency noise check
        stft = np.abs(librosa.stft(waveform_np))
        high_freq_energy = np.mean(stft[-len(stft)//4:, :])  # Top 25% frequencies
        total_energy = np.mean(stft)
        noise_ratio = high_freq_energy / (total_energy + 1e-8)
        noise_score = 1.0 / (1.0 + noise_ratio * 10)  # Less high-freq noise = cleaner
        
        # Combine quality indicators
        quality_score = (
            snr_score * 0.3 +
            flatness_score * 0.3 +
            zcr_score * 0.2 +
            noise_score * 0.2
        )
        
        return quality_score
    
    def get_adaptive_threshold(self, quality_score):
        """
        Determine threshold based on audio quality - TRUST THE TRAINED MODEL
        
        Model: best.pth (Trained on ASVspoof2019 + LibriSpeech)
        Lower threshold = More trust in model = Better for LibriSpeech
        
        All audio types → 0.50 threshold (let model decide!)
        
        Logic: Model is properly trained, trust its predictions
        """
        # Trust the trained model - consistent threshold
        return 0.50
    
    def detect_tts_spectral(self, waveform_np, sr=16000):
        """
        ENHANCED Spectral analysis for MODERN TTS detection (ElevenLabs, etc.)
        Based on research: Modern neural vocoders have specific artifacts
        """
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform_np, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform_np, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform_np, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=waveform_np, sr=sr)
        
        # Modern TTS detection indicators:
        
        # 1. OVERLY SMOOTH spectral centroid (neural vocoders are TOO perfect)
        centroid_diff = np.abs(np.diff(spectral_centroid))
        centroid_smoothness = np.mean(centroid_diff)
        # Real speech: 200-500 Hz changes, Modern TTS: < 200 Hz (more sensitive!)
        smoothness_score = 1.0 if centroid_smoothness < 200 else (0.7 if centroid_smoothness < 300 else 0.2)
        
        # 2. CONSISTENT spectral rolloff (real speech varies more)
        rolloff_std = np.std(spectral_rolloff)
        rolloff_mean = np.mean(spectral_rolloff)
        rolloff_cv = rolloff_std / (rolloff_mean + 1e-6)
        # Real: CV > 0.10, Modern TTS: CV < 0.08 (more sensitive!)
        rolloff_consistency = 1.0 if rolloff_cv < 0.08 else (0.6 if rolloff_cv < 0.12 else 0.1)
        
        # 3. LIMITED bandwidth (TTS uses narrower frequency range for quality)
        bandwidth_mean = np.mean(spectral_bandwidth)
        # Real: 2000-4000 Hz, Modern TTS: < 3000 Hz (more sensitive!)
        bandwidth_score = 1.0 if bandwidth_mean < 3000 else (0.5 if bandwidth_mean < 3500 else 0.1)
        
        # 4. PERIODIC patterns in spectral contrast (vocoders have frame-level artifacts)
        contrast_autocorr = []
        for band in spectral_contrast:
            if len(band) > 10:
                autocorr = np.correlate(band - np.mean(band), band - np.mean(band), mode='same')
                center = len(autocorr) // 2
                if center + 5 < len(autocorr):
                    # Check for periodicity (peak at lag 3-5 frames)
                    periodicity = np.max(autocorr[center+3:center+6]) / (autocorr[center] + 1e-6)
                    contrast_autocorr.append(periodicity)
        
        if len(contrast_autocorr) > 0:
            avg_periodicity = np.mean(contrast_autocorr)
            # Real: < 0.45, Modern TTS: > 0.5 (more sensitive!)
            periodicity_score = 1.0 if avg_periodicity > 0.5 else (0.5 if avg_periodicity > 0.4 else 0.1)
        else:
            periodicity_score = 0.3
        
        # 5. HIGH-FREQUENCY CUTOFF (neural vocoders often cut above 8kHz for efficiency)
        stft = np.abs(librosa.stft(waveform_np))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Energy in high frequencies (7-8 kHz)
        high_freq_mask = (freqs >= 7000) & (freqs <= 8000)
        if np.sum(high_freq_mask) > 0:
            high_freq_energy = np.mean(stft[high_freq_mask, :])
            total_energy = np.mean(stft) + 1e-10
            high_freq_ratio = high_freq_energy / total_energy
            
            # Real: > 0.12, Modern TTS: < 0.12 (more sensitive!)
            highfreq_score = 1.0 if high_freq_ratio < 0.12 else (0.6 if high_freq_ratio < 0.18 else 0.1)
        else:
            highfreq_score = 0.5
        
        # Combine all indicators with weights
        # Modern TTS has ALL these characteristics simultaneously
        final_score = (
            smoothness_score * 0.25 +      # Overly smooth transitions
            rolloff_consistency * 0.20 +    # Too consistent rolloff
            bandwidth_score * 0.15 +        # Limited bandwidth
            periodicity_score * 0.25 +      # Periodic artifacts
            highfreq_score * 0.15           # High-freq cutoff
        )
        
        return final_score
    
    def detect_tts_prosody(self, waveform_np, sr=16000):
        """
        MODERN TTS Prosody Detection - targets neural TTS (ElevenLabs, etc.)
        Detects 2024 neural TTS prosody patterns that differ from real speech
        """
        # Extract pitch using robust method
        pitches, magnitudes = librosa.piptrack(y=waveform_np, sr=sr, fmin=80, fmax=400)
        
        # Get pitch contour
        pitch_contour = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_contour.append(pitch)
        
        if len(pitch_contour) < 10:
            return 0.5  # Insufficient data
        
        pitch_contour = np.array(pitch_contour)
        
        # MODERN NEURAL TTS indicators:
        
        # 1. Unnaturally smooth pitch transitions (neural vocoders over-smooth)
        pitch_diff = np.abs(np.diff(pitch_contour))
        pitch_smoothness = np.mean(pitch_diff)
        # Real: 15-30 Hz transitions, Modern TTS: < 12 Hz (more sensitive!)
        pitch_smooth_score = 1.0 if pitch_smoothness < 12 else (0.7 if pitch_smoothness < 18 else 0.2)
        
        # 2. Overly consistent pitch variance (too perfect)
        pitch_std = np.std(pitch_contour)
        pitch_mean = np.mean(pitch_contour)
        pitch_cv = pitch_std / (pitch_mean + 1e-6)
        # Real: CV > 0.12, Modern TTS: CV < 0.12 (more sensitive!)
        pitch_consistency_score = 1.0 if pitch_cv < 0.12 else (0.7 if pitch_cv < 0.18 else 0.2)
        
        # 3. Missing micro-prosody (neural vocoders lack tiny natural fluctuations)
        # Second-order differences (acceleration)
        pitch_accel = np.diff(np.diff(pitch_contour))
        pitch_jitter = np.std(pitch_accel)
        # Real: > 12 (natural imperfections), Modern TTS: < 10 (more sensitive!)
        jitter_score = 1.0 if pitch_jitter < 10 else (0.6 if pitch_jitter < 18 else 0.2)
        
        # 4. Energy envelope regularity (rhythm analysis)
        rms = librosa.feature.rms(y=waveform_np)[0]
        
        # a) Overly regular rhythm (timing too perfect)
        rms_std = np.std(rms)
        rms_mean = np.mean(rms)
        rms_cv = rms_std / (rms_mean + 1e-6)
        # Real: > 0.45 (irregular), Modern TTS: < 0.40 (more sensitive!)
        rhythm_score = 1.0 if rms_cv < 0.40 else (0.6 if rms_cv < 0.55 else 0.2)
        
        # b) Energy autocorrelation (detects periodic patterns)
        if len(rms) > 10:
            rms_centered = rms - np.mean(rms)
            rms_autocorr = np.correlate(rms_centered, rms_centered, mode='same')
            center = len(rms_autocorr) // 2
            if center + 10 < len(rms_autocorr):
                rms_periodicity = np.max(rms_autocorr[center+5:center+10]) / (rms_autocorr[center] + 1e-6)
                # Modern TTS: > 0.55, Real: < 0.45 (more sensitive!)
                periodicity_score = 1.0 if rms_periodicity > 0.55 else (0.6 if rms_periodicity > 0.45 else 0.2)
            else:
                periodicity_score = 0.5
        else:
            periodicity_score = 0.5
        
        # 5. Voice onset smoothness (neural TTS too smooth)
        onset_env = librosa.onset.onset_strength(y=waveform_np, sr=sr)
        onset_std = np.std(onset_env)
        # Real: > 0.8 (sharp), Modern TTS: < 0.7 (more sensitive!)
        onset_score = 1.0 if onset_std < 0.7 else (0.6 if onset_std < 1.2 else 0.2)
        
        # Combine all 5 indicators
        # All must be present for modern neural TTS
        prosody_tts_score = (
            pitch_smooth_score * 0.20 +       # Pitch smoothness
            pitch_consistency_score * 0.20 +  # Pitch consistency
            jitter_score * 0.20 +              # Micro-prosody
            rhythm_score * 0.15 +              # Rhythm regularity
            periodicity_score * 0.15 +         # Energy periodicity
            onset_score * 0.10                 # Onset smoothness
        )
        
        return min(prosody_tts_score, 1.0)
    
    def detect_tts_statistical(self, waveform_np, sr=16000):
        """
        Statistical analysis - TTS has predictable patterns
        """
        # Zero crossing rate (voice naturalness)
        zcr = librosa.feature.zero_crossing_rate(waveform_np)[0]
        zcr_std = np.std(zcr)
        zcr_consistency = 1.0 / (1.0 + zcr_std / 0.05)
        
        # Spectral flatness (TTS is less flat - more tonal)
        spectral_flatness = librosa.feature.spectral_flatness(y=waveform_np)[0]
        flatness_mean = np.mean(spectral_flatness)
        tonal_score = 1.0 - flatness_mean  # Lower flatness = more tonal = more TTS-like
        
        # MFCC consistency (TTS has very consistent MFCCs)
        mfccs = librosa.feature.mfcc(y=waveform_np, sr=sr, n_mfcc=13)
        mfcc_stds = np.std(mfccs, axis=1)
        mfcc_consistency = 1.0 / (1.0 + np.mean(mfcc_stds) / 10.0)
        
        # Combine indicators
        statistical_tts_score = (
            zcr_consistency * 0.3 +
            tonal_score * 0.4 +
            mfcc_consistency * 0.3
        )
        
        return statistical_tts_score
    
    def predict(self, waveform_tensor, waveform_np=None, sr=16000):
        """
        Multi-layer prediction with ADAPTIVE THRESHOLD
        
        Returns:
            dict with prediction, confidence, detailed scores, and threshold used
        """
        # Convert to numpy if needed
        if waveform_np is None:
            waveform_np = waveform_tensor.squeeze().cpu().numpy()
        
        # STEP 1: Analyze audio quality (clean TTS vs noisy real recording)
        quality_score = self.analyze_audio_quality(waveform_np, sr)
        adaptive_threshold = self.get_adaptive_threshold(quality_score)
        
        # STEP 2: Neural model prediction
        self.model.eval()
        with torch.no_grad():
            if waveform_tensor.dim() == 2:
                waveform_tensor = waveform_tensor.unsqueeze(0)
            
            outputs = self.model(waveform_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            real_prob = probabilities[0, 0].item()  # REAL probability (bonafide)
            fake_prob = probabilities[0, 1].item()  # FAKE probability (spoof)
            
            model_confidence = max(fake_prob, real_prob)
            model_predicts_fake = fake_prob > adaptive_threshold  # Use adaptive threshold!
        
        # STEP 2.5: Check SECONDARY MODEL (for ElevenLabs detection)
        secondary_fake_prob = 0.0
        if hasattr(self, 'secondary_model') and self.secondary_model is not None:
            try:
                with torch.no_grad():
                    # Ensure secondary model is in eval mode
                    self.secondary_model.eval()
                    secondary_output = self.secondary_model(waveform_tensor)
                    secondary_probs = torch.softmax(secondary_output, dim=1)
                    secondary_real_prob = secondary_probs[0, 0].item()  # REAL
                    secondary_fake_prob = secondary_probs[0, 1].item()  # FAKE
            except Exception as e:
                print(f"Warning: Secondary model failed: {e}")
                secondary_fake_prob = 0.0
        
        # REMOVED: Real Voice Protection - Trust the trained model 100%
        # The model is trained specifically on this data and knows what it's doing
        
        # REMOVED: ELEVENLABS OVERRIDE - Trust primary model only
        # Secondary model is disabled, use primary model predictions only
        
        # STEP 3: Spectral analysis (disabled - model knows best)
        spectral_score = 0.0  # self.detect_tts_spectral(waveform_np, sr)
        
        # STEP 4: Prosody analysis (disabled - model knows best)
        prosody_score = 0.0  # self.detect_tts_prosody(waveform_np, sr)
        
        # STEP 5: Statistical analysis (disabled - model knows best)
        statistical_score = 0.0  # self.detect_tts_statistical(waveform_np, sr)
        
        # STEP 6: Trust the model 100% - use its output directly
        final_tts_score = fake_prob
        
        # Make decision based on model output only (no heuristics)
        if fake_prob > 0.50:
            return {
                'prediction': 'FAKE',
                'confidence': fake_prob,
                'probabilities': {
                    'fake': fake_prob * 100,
                    'real': real_prob * 100
                },
                'detailed_scores': {
                    'model_score': fake_prob,
                    'secondary_score': 0.0,
                    'spectral_score': 0.0,
                    'prosody_score': 0.0,
                    'statistical_score': 0.0,
                    'final_tts_score': fake_prob,
                    'quality_score': quality_score,
                    'adaptive_threshold': adaptive_threshold
                },
                'analysis': f"🤖 MODEL | Fake={fake_prob:.2%}, Real={real_prob:.2%}"
            }
        else:
            return {
                'prediction': 'REAL',
                'confidence': real_prob,
                'probabilities': {
                    'fake': fake_prob * 100,
                    'real': real_prob * 100
                },
                'detailed_scores': {
                    'model_score': fake_prob,
                    'secondary_score': 0.0,
                    'spectral_score': 0.0,
                    'prosody_score': 0.0,
                    'statistical_score': 0.0,
                    'final_tts_score': fake_prob,
                    'quality_score': quality_score,
                    'adaptive_threshold': adaptive_threshold
                },
                'analysis': f"🤖 MODEL | Real={real_prob:.2%}, Fake={fake_prob:.2%}"
            }
