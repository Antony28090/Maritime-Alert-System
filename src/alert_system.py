import os
import time
from gtts import gTTS
try:
    from playsound import playsound
except ImportError:
    # Fallback if playsound has issues (common on some envs), or mock it
    def playsound(f):
        print(f"[AUDIO] Playing {f} (simulated)")

class AlertSystem:
    def __init__(self):
        self.audio_dir = 'data/audio'
        os.makedirs(self.audio_dir, exist_ok=True)
        self.last_alert_time = 0
        self.alert_cooldown = 10 # Seconds between alerts
        
        self.generate_audio_files()
        
    def generate_audio_files(self):
        # Tamil Alert Messages
        messages = {
            'danger': "எச்சரிக்கை! நீங்கள் எல்லை தாண்டும் அபாயத்தில் உள்ளீர்கள். உடனே திரும்புங்கள்.",
            'caution': "கவனிக்கவும். நீங்கள் எல்லையை நெருங்குகிறீர்கள்."
        }
        
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
                # playsound 1.2.2 (common legacy ver) has issues with spaces in paths
                # os.startfile is a robust Windows native alternative that opens the default player
                # but might pop up a window.
                # Let's try fixing playsound first with absolute path.
                playsound(path)
            except Exception as e:
                print(f"Error playing audio: {e}")
                # Fallback to system beep if audio fails
                import winsound
                winsound.Beep(1000, 500) # Freq, Duration
