import time

from topic_bandit import TopicBandit


def test_topic_bandit_basic_flow():
    topics = ["hobby", "food", "travel", "music", "movies"]
    bandit = TopicBandit(topics, alpha=0.8)
    bandit.enable_llm_reward = False

    emotion = {"intensity": 0.7, "confidence": 0.6, "primary_emotions": ["joy"]}
    context = "User enjoyed travel and food discussions recently."

    topic_idx, selected_topic = bandit.select_topic(context=context, emotion=emotion)
    assert 0 <= topic_idx < len(topics)
    assert selected_topic in topics

    response = "That sounds fun! 次はどんな旅を計画していますか？"
    user_input = "I love planning new trips every spring."
    reward = bandit.evaluate_response(response, user_input, emotion=emotion)
    assert 0.0 <= reward <= 1.0

    bandit.update(topic_idx, reward)
    bandit.add_to_history(user_input, response, selected_topic)

    stats = bandit.get_stats()
    assert selected_topic in stats
    assert stats[selected_topic]["count"] >= 1
    assert 0.0 <= stats[selected_topic]["avg_reward"] <= 1.0

    # ensure recency feature changes over time
    before = bandit._recency_score(topic_idx, time.time())
    time.sleep(0.01)
    after = bandit._recency_score(topic_idx, time.time())
    assert after >= before
