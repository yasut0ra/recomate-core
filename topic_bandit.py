import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import openai

class LinUCBBandit:
    def __init__(self, n_arms: int, feature_dim: int, alpha: float = 1.2, lambda_param: float = 1.0):
        self.n_arms = n_arms
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_param = lambda_param

        self.A = [lambda_param * np.eye(feature_dim) for _ in range(n_arms)]
        self.b = [np.zeros(feature_dim) for _ in range(n_arms)]

    def select(self, feature_map: Dict[int, np.ndarray]) -> int:
        best_idx = 0
        best_score = -np.inf

        for idx, features in feature_map.items():
            if features.shape[0] != self.feature_dim:
                raise ValueError("Feature dimension mismatch in LinUCBBandit")

            theta = np.linalg.solve(self.A[idx], self.b[idx])
            A_inv_x = np.linalg.solve(self.A[idx], features)

            exploit = float(theta @ features)
            explore = float(np.sqrt(features @ A_inv_x))
            score = exploit + self.alpha * explore

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def update(self, arm_index: int, reward: float, features: np.ndarray) -> None:
        if features.shape[0] != self.feature_dim:
            raise ValueError("Feature dimension mismatch in LinUCBBandit.update")

        reward = float(np.clip(reward, 0.0, 1.0))
        self.A[arm_index] += np.outer(features, features)
        self.b[arm_index] += reward * features

    def get_expected_reward(self, arm_index: int, features: np.ndarray) -> float:
        theta = np.linalg.solve(self.A[arm_index], self.b[arm_index])
        return float(theta @ features)


class TopicBandit:
    def __init__(
        self,
        topics: Iterable[str],
        alpha: float = 1.2,
        lambda_param: float = 1.0,
        recency_halflife: float = 120.0,
    ) -> None:
        self.topics: List[str] = list(topics)
        if not self.topics:
            raise ValueError("Topic list must not be empty")

        self.n_topics = len(self.topics)
        self.feature_dim = 6
        self.strategy = LinUCBBandit(self.n_topics, self.feature_dim, alpha=alpha, lambda_param=lambda_param)

        self.recency_halflife = max(1.0, recency_halflife)
        self.topic_counts = np.zeros(self.n_topics, dtype=np.int64)
        self.topic_reward_sums = np.zeros(self.n_topics, dtype=np.float64)
        self.topic_last_timestamp = np.zeros(self.n_topics, dtype=np.float64)
        self.last_selected_index: Optional[int] = None
        self.last_features: Dict[int, np.ndarray] = {}
        self.conversation_history: List[Dict] = []

        openai.api_key = openai.api_key or os.getenv("OPENAI_API_KEY")
        self.llm_model = os.getenv("OPENAI_CHAT_COMPLETION_MODEL", "gpt-3.5-turbo")
        self.enable_llm_reward = os.getenv("USE_LLM_REWARD", "0") == "1"

    def select_topic(
        self,
        epsilon: float = 0.0,
        context: str | None = None,
        emotion: Optional[Dict] = None,
    ) -> Tuple[int, str]:
        now = time.time()
        context_str = context or ""
        emotion_data = emotion or {}

        feature_map = self._build_feature_map(context_str, emotion_data, now)

        if epsilon > 0.0 and np.random.random() < epsilon:
            topic_idx = int(np.random.randint(self.n_topics))
        else:
            topic_idx = self.strategy.select(feature_map)

        self.last_selected_index = topic_idx
        self.last_features[topic_idx] = feature_map[topic_idx]

        return topic_idx, self.topics[topic_idx]

    def evaluate_response(
        self,
        response: str,
        user_input: str,
        emotion: Optional[Dict] = None,
    ) -> float:
        if not response.strip():
            return 0.0

        length_score = min(len(response) / 120.0, 1.0)
        question_bonus = 0.2 if ("?" in response or "？" in response) else 0.0
        empathy_bonus = 0.0
        if emotion:
            for primary in emotion.get("primary_emotions", []):
                if primary and primary in response:
                    empathy_bonus = 0.2
                    break

        diversity_penalty = 0.0
        if self.last_selected_index is not None and self.topic_counts[self.last_selected_index] > 0:
            diversity_penalty = 0.05 * max(0, 3 - self._time_since_last(self.last_selected_index))

        heuristic_score = np.clip(0.5 * length_score + question_bonus + empathy_bonus - diversity_penalty, 0.0, 1.0)

        if not self.enable_llm_reward:
            return float(heuristic_score)

        try:
            prompt = f"""
            以下のユーザー入力と VTuber の応答を 0.0 から 1.0 の範囲で評価してください。

            ユーザー入力: {user_input}
            VTuber の応答: {response}

            観点は以下の通りです。
            - 共感や気遣いがあるか
            - 会話を広げる工夫があるか
            - 不自然な点がないか

            数値のみを出力してください。
            """
            completion = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは会話の質を評価する厳格なレビュアーです。数値のみを返してください。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            score_text = completion.choices[0].message.content.strip()
            llm_score = float(np.clip(float(score_text), 0.0, 1.0))
        except Exception as exc:  # noqa: BLE001
            print(f"応答評価でエラーが発生しました: {exc}")
            llm_score = heuristic_score

        return float(0.5 * heuristic_score + 0.5 * llm_score)

    def generate_subtopics(self, main_topic: str) -> List[str]:
        prompt = f"""
        トピック「{main_topic}」について会話を広げるためのサブトピックを 3 つ提案してください。
        箇条書きで短く出力してください。
        """
        try:
            completion = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "あなたはVTuberの会話を支援するアシスタントです。具体的で親しみやすいサブトピックを提案してください。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            lines = [line.strip("- ・") for line in completion.choices[0].message.content.splitlines() if line.strip()]
            return lines[:3]
        except Exception as exc:  # noqa: BLE001
            print(f"サブトピック生成でエラーが発生しました: {exc}")
            return []

    def update(
        self,
        topic_idx: int,
        reward: float,
        features: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        if features is None:
            features = self.last_features.get(topic_idx)
        if features is None:
            features = self._build_feature_map("", {}, time.time())[topic_idx]

        self.strategy.update(topic_idx, reward, features)
        self.topic_counts[topic_idx] += 1
        self.topic_reward_sums[topic_idx] += float(np.clip(reward, 0.0, 1.0))
        self.topic_last_timestamp[topic_idx] = timestamp or time.time()
        self.last_features[topic_idx] = features

    def add_to_history(self, user_input: str, response: str, topic: str) -> None:
        self.conversation_history.append(
            {
                "user_input": user_input,
                "response": response,
                "topic": topic,
                "timestamp": time.time(),
            }
        )

    def get_topic_stats(self) -> Dict[str, Dict[str, float]]:
        default_features = self._build_feature_map("", {}, time.time())
        stats: Dict[str, Dict[str, float]] = {}
        for idx, topic in enumerate(self.topics):
            stats[topic] = {
                "count": float(self.topic_counts[idx]),
                "avg_reward": float(
                    self.topic_reward_sums[idx] / self.topic_counts[idx]
                    if self.topic_counts[idx] > 0
                    else 0.0
                ),
                "expected_reward": self.strategy.get_expected_reward(idx, default_features[idx]),
            }
        return stats

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        return self.get_topic_stats()

    def _build_feature_map(
        self,
        context: str,
        emotion: Dict,
        now: float,
    ) -> Dict[int, np.ndarray]:
        context_lower = context.lower()
        intensity = float(emotion.get("intensity", 0.5))
        confidence = float(emotion.get("confidence", 0.5))

        feature_map: Dict[int, np.ndarray] = {}
        for idx, topic in enumerate(self.topics):
            novelty = 1.0 / (1.0 + self.topic_counts[idx])
            recency_feature = self._recency_score(idx, now)
            topic_in_context = 1.0 if topic.lower() in context_lower else 0.0

            features = np.array(
                [
                    1.0,
                    intensity,
                    confidence,
                    recency_feature,
                    novelty,
                    topic_in_context,
                ],
                dtype=np.float64,
            )
            feature_map[idx] = features

        return feature_map

    def _recency_score(self, topic_idx: int, now: float) -> float:
        last_time = self.topic_last_timestamp[topic_idx]
        if last_time <= 0.0:
            return 1.0
        elapsed = max(0.0, now - last_time)
        return float(elapsed / (elapsed + self.recency_halflife))

    def _time_since_last(self, topic_idx: int) -> float:
        last_time = self.topic_last_timestamp[topic_idx]
        if last_time <= 0.0:
            return float("inf")
        return time.time() - last_time
