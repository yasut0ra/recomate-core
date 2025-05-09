from topic_bandit import TopicBandit
import time

def test_topic_bandit():
    """トピック選択システムのテスト"""
    # テスト用のトピック
    topics = [
        "趣味",
        "食べ物",
        "旅行",
        "音楽",
        "映画"
    ]
    
    bandit = TopicBandit(topics)
    
    print("トピック選択システムのテストを開始します...")
    
    try:
        # トピック選択のテスト
        print("\nトピック選択のテスト:")
        for _ in range(5):
            topic_idx, selected_topic = bandit.select_topic()
            print(f"選択されたトピック: {selected_topic}")
            time.sleep(0.5)
        
        # 報酬の更新テスト
        print("\n報酬の更新テスト:")
        for i in range(len(topics)):
            reward = 0.8 if i % 2 == 0 else 0.3  # 交互に高い/低い報酬
            bandit.update(i, reward)
            print(f"トピック '{topics[i]}' の報酬を {reward} に更新")
        
        # 統計情報の表示
        print("\nトピックの統計情報:")
        stats = bandit.get_stats()
        for topic, stat in stats.items():
            print(f"トピック '{topic}':")
            print(f"- 選択回数: {stat['count']}")
            print(f"- 平均報酬: {stat['avg_reward']:.3f}")
            print(f"- 期待報酬: {stat['expected_reward']:.3f}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    test_topic_bandit() 