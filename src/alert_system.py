import os
import time
from gtts import gTTS
import pygame

class AlertSystem:
    def __init__(self):
        self.audio_dir = 'data/audio'
        os.makedirs(self.audio_dir, exist_ok=True)
        self.last_alert_time = 0
        self.alert_cooldown = 10 # Seconds between alerts
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"Warning: Audio mixer init failed: {e}")
        
        self.generate_audio_files()
        
    def generate_audio_files(self):
        # Tamil Alert Messages
        messages = {
            'crossed': "எச்சரிக்கை! நீங்கள் எல்லையைத் தாண்டிவிட்டீர்கள். உடனடியாகத் திரும்பிச் செல்லவும்.", # "Warning! You have crossed... Turn back..."
            'danger': "எச்சரிக்கை! நீங்கள் எல்லை தாண்டும் அபாயத்தில் உள்ளீர்கள். உடனே திரும்புங்கள்.", # "Warning! You are at risk... Turn back..."
            'caution': "கவனிக்கவும். நீங்கள் எல்லையை நெருங்குகிறீர்கள்."
        }
        
        # Remove old files if message changed
        if os.path.exists(self.audio_dir):
            import shutil
            shutil.rmtree(self.audio_dir)
        os.makedirs(self.audio_dir, exist_ok=True)
        
        for level, text in messages.items():
            path = os.path.join(self.audio_dir, f"{level}.mp3")
            if not os.path.exists(path):
                print(f"Generating audio for {level}...")
                tts = gTTS(text=text, lang='ta')
                tts.save(path)
                
    def trigger_alert(self, level):
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
            
        path = os.path.abspath(os.path.join(self.audio_dir, f"{level}.mp3"))
        if os.path.exists(path):
            print(f"\n*** ALERT ({level.upper()}) TRIGGERED ***")
            try:
                # Use pygame for stable playback
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                
                # We could wait, but sticking to non-blocking is better for simulation loop
                # However, if we don't wait or keep reference, it might cut off if object destroyed?
                # Pygame mixer runs in background thread usually.
                
                self.last_alert_time = current_time
            except Exception as e:
                print(f"Error playing audio: {e}")
                # Fallback to system beep if audio fails
                import winsound
                winsound.Beep(1000, 500) # Freq, Duration

    def check_zone(self, lat, lon):
        """
        Determines the zone based on distance from IMBL.
        """
        from src.config import IMBL_POINTS, DANGER_DIST_KM, CAUTION_DIST_KM
        from src.geometry import distance_from_polyline, is_sri_lankan_side
        
        dist, _ = distance_from_polyline([lat, lon], IMBL_POINTS)
        is_sl = is_sri_lankan_side([lat, lon], IMBL_POINTS)
        
        if is_sl:
            return "DANGER"
        elif dist < DANGER_DIST_KM:
            return "DANGER"
        elif dist < CAUTION_DIST_KM:
            return "CAUTION"
        else:
            return "SAFE"
