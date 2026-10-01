# NaVIP Spoken Voice Navigation System

This module generates spoken voice guidance and NaVIP-style clock-position announcements.

## Components
- `instruction_generator.py`: Computes turn directions ("Turn left", "Continue straight") and relative clock positions ("[POI name] is at 2 o'clock, 1.5m away").
- `voice_output.py`: Spoken output manager. Always prints text instructions to console; uses Windows PowerShell SAPI or `pyttsx3` for TTS audio playback.

## Example Spoken Output
- `"Move forward for 1.2 meters."`
- `"Turn left and proceed for 0.8 meters."`
- `"[Queen Side Board Corner] is at your 2 o'clock position, 1.5 meters away."`
- `"Destination reached."`
