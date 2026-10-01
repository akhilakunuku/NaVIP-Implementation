"""
voice_output.py - Modular Voice Output Engine

Prints natural voice instructions to stdout and attempts optional text-to-speech (TTS) output.
Supports Windows SAPI (via win32com or PowerShell) and pyttsx3 with graceful fallback.
If TTS is unavailable, the application functions normally without crashing.
"""

import sys
import subprocess


class VoiceOutputManager:
    def __init__(self, enable_tts=True):
        self.enable_tts = enable_tts
        self.tts_engine = None

        if self.enable_tts:
            self._init_tts_engine()

    def _init_tts_engine(self):
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
        except Exception:
            self.tts_engine = None

    def speak(self, text_instruction):
        """
        Outputs instruction to console and speaks via TTS if available.
        """
        print(f"\n[VOICE GUIDANCE]: \"{text_instruction}\"")

        if not self.enable_tts:
            return

        # Attempt pyttsx3 first
        if self.tts_engine is not None:
            try:
                self.tts_engine.say(text_instruction)
                self.tts_engine.runAndWait()
                return
            except Exception:
                pass

        # Fallback to Windows PowerShell SAPI SpeechSynthesizer
        try:
            ps_cmd = (
                f'Add-Type -AssemblyName System.Speech; '
                f'$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$synth.Speak("{text_instruction}");'
            )
            subprocess.run(["powershell", "-Command", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except Exception:
            # SAPI fallback unavailable; text output already printed above
            pass


if __name__ == "__main__":
    voice = VoiceOutputManager(enable_tts=True)
    print("==================================================")
    print("Voice Output Engine Test")
    print("==================================================")
    voice.speak("Destination reached. You have arrived at Observation Desk.")
    print("==================================================")
