#!/usr/bin/env python3
"""
Synthesize alert sound for crypto trading alerts.
Run this script first to create the alert_sound.wav file.
"""

import numpy as np
import wave
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def synthesize_alert_sound():
    """
    Synthesize a loud alert sound and save as .wav file
    """
    # Sound parameters
    sample_rate = 44100  # Hz
    duration = 2.0  # seconds
    frequency = 800  # Hz (alert tone)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a loud alert sound with multiple frequencies
    # Main tone
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Add harmonics for a more attention-grabbing sound
    signal += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)  # Second harmonic
    signal += 0.3 * np.sin(2 * np.pi * frequency * 3 * t)  # Third harmonic
    
    # Add a sweep effect
    sweep_freq = np.linspace(frequency, frequency * 1.5, len(t))
    sweep_signal = np.sin(2 * np.pi * sweep_freq * t)
    signal += 0.3 * sweep_signal
    
    # Normalize and make it loud
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Convert to 16-bit integers
    signal_int = (signal * 32767).astype(np.int16)
    
    # Save as WAV file
    filename = "alert_sound.wav"
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal_int.tobytes())
    
    logger.info(f"Alert sound synthesized and saved as {filename}")
    return filename

def main():
    """Main function to synthesize the alert sound"""
    logger.info("Starting alert sound synthesis...")
    
    # Check if sound file already exists
    alert_sound_file = "alert_sound.wav"
    if os.path.exists(alert_sound_file):
        logger.info(f"Alert sound file {alert_sound_file} already exists.")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            logger.info("Keeping existing sound file.")
            return
    
    # Synthesize the alert sound
    try:
        filename = synthesize_alert_sound()
        logger.info(f"✅ Alert sound successfully created: {filename}")
        logger.info("You can now run btc_hourly_alert.py")
    except Exception as e:
        logger.error(f"❌ Error synthesizing alert sound: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 