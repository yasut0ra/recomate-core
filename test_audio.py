import sounddevice as sd
import numpy as np

def print_audio_devices():
    """利用可能な音声デバイスを表示"""
    print("利用可能な音声デバイス:")
    print(sd.query_devices())

def test_audio_input():
    """音声入力のテスト"""
    # デフォルトのデバイスを取得
    device_info = sd.query_devices(kind='input')
    samplerate = int(device_info['default_samplerate'])
    
    # 録音時間（秒）
    duration = 5
    
    print(f"録音を開始します... {duration}秒間話してください")
    
    # 録音
    recording = sd.rec(int(samplerate * duration), 
                      samplerate=samplerate,
                      channels=1, 
                      dtype=np.float32)
    sd.wait()
    
    print("録音が完了しました")
    
    # 音量レベルを確認
    volume = np.abs(recording).mean()
    print(f"平均音量レベル: {volume:.6f}")

if __name__ == "__main__":
    print_audio_devices()
    print("\n" + "="*50 + "\n")
    test_audio_input() 