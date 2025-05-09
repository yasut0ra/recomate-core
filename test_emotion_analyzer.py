from emotion_analyzer import EmotionAnalyzer
import time

def test_emotion_analyzer():
    """感情分析システムのテスト"""
    analyzer = EmotionAnalyzer()
    
    # テスト用のテキスト
    test_texts = [
        "今日はとても楽しい一日でした！",
        "最近、少し寂しい気分です。",
        "それは本当に腹立たしいですね。",
        "えっ！？本当ですか！？",
        "今日は普通の一日でした。"
    ]
    
    print("感情分析のテストを開始します...")
    
    try:
        for text in test_texts:
            print(f"\nテキスト: {text}")
            
            # 感情分析の実行
            emotion_data = analyzer.analyze_emotion(text)
            
            # 結果の表示
            print("分析結果:")
            print(f"- 主要な感情: {', '.join(emotion_data['primary_emotions'])}")
            print(f"- 感情の強度: {emotion_data['intensity']}")
            print(f"- 感情の組み合わせ: {emotion_data['emotion_combination']}")
            print(f"- 感情の変化: {emotion_data['emotion_change']}")
            
            # 感情表現の生成
            expression = analyzer.get_emotion_expression(emotion_data)
            print(f"生成された表情: {expression}")
            
            time.sleep(1)  # 結果を確認しやすくするため
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    test_emotion_analyzer() 