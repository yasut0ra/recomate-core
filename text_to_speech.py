from gtts import gTTS
import tempfile
import os
import pygame

class TextToSpeech:
    def __init__(self, language='ja'):
        self.language = language
        pygame.mixer.init()
        
    def speak(self, text):
        """テキストを音声に変換して再生"""
        try:
            # 一時ファイルを作成
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_filename = temp_file.name
            
            # テキストを音声に変換
            tts = gTTS(text=text, lang=self.language)
            tts.save(temp_filename)
            
            # 音声を再生
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # 一時ファイルを削除
            os.unlink(temp_filename)
            
        except Exception as e:
            print(f"音声合成エラー: {e}")
            
    def change_language(self, language):
        """言語を変更"""
        self.language = language

if __name__ == "__main__":
    # テスト用
    tts = TextToSpeech()
    tts.speak("こんにちは、私はAI Vtuberです。") 