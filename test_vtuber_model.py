from vtuber_model import VtuberModel
import time

def test_vtuber_model():
    """VTuberモデルのテスト"""
    model = VtuberModel()
    
    # 各表情のテスト
    expressions = ['neutral', 'happy', 'sad', 'angry', 'surprised']
    
    try:
        for expression in expressions:
            print(f"表情を {expression} に変更します")
            model.update(emotion=expression)
            model.render()
            time.sleep(2)  # 2秒間表示
        
        # 口パクアニメーションのテスト
        print("口パクアニメーションをテストします")
        for _ in range(30):  # 30フレーム分
            model.update(is_speaking=True)
            model.render()
            time.sleep(0.1)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        model.cleanup()

if __name__ == "__main__":
    test_vtuber_model() 