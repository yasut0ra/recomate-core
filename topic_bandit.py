import os
import re
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

            theta = self._solve(self.A[idx], self.b[idx])
            A_inv_x = self._solve(self.A[idx], features)

            exploit = float(theta @ features)
            explore = float(np.sqrt(max(features @ A_inv_x, 0.0)))
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
        theta = self._solve(self.A[arm_index], self.b[arm_index])
        return float(theta @ features)

    @staticmethod
    def _solve(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix) @ vector


class TopicBandit:
    NEGATIVE_KEYWORDS = (
        "興味ない",
        "興味がない",
        "以外",
        "違う",
        "ちがう",
        "別の",
        "他の",
        "やめて",
        "飽きた",
        "飽き",
        "嫌い",
        "嫌",
        "いらない",
        "やだ",
        "no more",
        "not interested",
        "stop",
    )
    PIVOT_KEYWORDS = (
        "別の",
        "他の話",
        "違う話",
        "変えて",
        "変わ",
        "切り替えて",
        "もっと別",
        "ほかの",
        "pivot",
        "switch",
    )
    POSITIVE_KEYWORDS = (
        "好き",
        "もっと",
        "興味ある",
        "面白い",
        "楽しい",
        "嬉しい",
        "ありがとう",
        "good",
        "great",
    )

    TOKEN_PATTERN = re.compile(r"[\wぁ-んァ-ン一-龯]+")

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
        self.feature_dim = 8
        self.strategy = LinUCBBandit(self.n_topics, self.feature_dim, alpha=alpha, lambda_param=lambda_param)

        self.recency_halflife = max(1.0, recency_halflife)
        self.topic_counts = np.zeros(self.n_topics, dtype=np.int64)
        self.topic_reward_sums = np.zeros(self.n_topics, dtype=np.float64)
        self.topic_last_timestamp = np.zeros(self.n_topics, dtype=np.float64)
        self.topic_negative_score = np.zeros(self.n_topics, dtype=np.float64)
        self.topic_index = {topic: idx for idx, topic in enumerate(self.topics)}

        self.previous_topic_index: Optional[int] = None
        self.last_selected_index: Optional[int] = None
        self.just_pivoted = False

        self.last_features: Dict[int, np.ndarray] = {}
        self.conversation_history: List[Dict] = []

        self.pending_pivot_pressure = 0.0
        self.last_negative_feedback = 0.0
        self._last_decay_time = time.time()
        self._last_question_count = 0
        self._last_response_length = 0
        self.last_reward_value = 0.0

        openai.api_key = openai.api_key or os.getenv("OPENAI_API_KEY")
        self.llm_model = os.getenv("OPENAI_CHAT_COMPLETION_MODEL", "gpt-3.5-turbo")
        self.enable_llm_reward = os.getenv("USE_LLM_REWARD", "0") == "1"

    def observe_feedback(self, user_text: str) -> None:
        if not user_text:
            self.pending_pivot_pressure *= 0.8
            self.last_negative_feedback *= 0.5
            return

        severity = self._negativity_score(user_text)
        pivot = self._pivot_intent_score(user_text)
        self.pending_pivot_pressure = max(self.pending_pivot_pressure * 0.6, pivot)
        self.last_negative_feedback = max(self.last_negative_feedback * 0.5, severity)

        if self.last_selected_index is not None:
            if severity > 0.0:
                self.topic_negative_score[self.last_selected_index] = min(
                    1.0,
                    0.5 * self.topic_negative_score[self.last_selected_index] + severity,
                )
            else:
                self.topic_negative_score[self.last_selected_index] *= 0.85

    def select_topic(
        self,
        epsilon: float = 0.0,
        context: str | None = None,
        emotion: Optional[Dict] = None,
    ) -> Tuple[int, str]:
        now = time.time()
        self._apply_decay(now)

        context_str = context or ""
        emotion_data = emotion or {}
        feature_map = self._build_feature_map(context_str, emotion_data, now)

        if epsilon > 0.0 and np.random.random() < epsilon:
            topic_idx = int(np.random.randint(self.n_topics))
        else:
            topic_idx = self.strategy.select(feature_map)

        self.previous_topic_index = self.last_selected_index
        self.just_pivoted = self.previous_topic_index is not None and self.previous_topic_index != topic_idx
        self.last_selected_index = topic_idx
        self.last_features[topic_idx] = feature_map[topic_idx]

        if self.just_pivoted:
            self.pending_pivot_pressure *= 0.5

        return topic_idx, self.topics[topic_idx]

    def evaluate_response(
        self,
        response: str,
        user_input: str,
        emotion: Optional[Dict] = None,
    ) -> float:
        response = (response or "").strip()
        if not response:
            self._last_response_length = 0
            self._last_question_count = 0
            return 0.0

        negativity_level = self._negativity_score(user_input)
        pivot_request = self._pivot_intent_score(user_input)

        response_length = len(response)
        self._last_response_length = response_length
        length_score = float(np.exp(-((response_length - 90.0) ** 2) / (2 * 45.0 ** 2)))
        length_score = np.clip(length_score, 0.0, 1.0)

        question_count = response.count("?") + response.count("？")
        self._last_question_count = question_count
        if question_count == 0:
            question_score = 0.35
        elif question_count == 1:
            question_score = 0.9
        else:
            question_score = max(0.2, 0.7 - 0.2 * (question_count - 1))

        empathy_score = 0.0
        if emotion:
            primaries = emotion.get("primary_emotions") or []
            matches = sum(1 for p in primaries if p and p in response)
            empathy_score = np.clip(0.2 * matches + 0.3 * float(emotion.get("intensity", 0.0)), 0.0, 1.0)
        elif any(token in response for token in ("ごめん", "すみません", "申し訳")):
            empathy_score = 0.4

        overlap_score = self._overlap_score(user_input, response)

        repetition_penalty = 0.0
        if not self.just_pivoted and self.previous_topic_index is not None and self.previous_topic_index == self.last_selected_index:
            base_penalty = 0.1
            if self.last_selected_index is not None:
                base_penalty += 0.2 * float(self.topic_negative_score[self.last_selected_index])
            repetition_penalty += base_penalty

        if pivot_request > 0.0 and not self.just_pivoted:
            repetition_penalty += 0.25 * pivot_request

        if negativity_level > 0.0 and not self.just_pivoted:
            repetition_penalty += 0.2 * negativity_level

        pivot_bonus = 0.0
        if pivot_request > 0.0 and self.just_pivoted:
            pivot_bonus = 0.12 * pivot_request

        heuristic_score = np.clip(
            0.45 * length_score
            + 0.2 * question_score
            + 0.2 * overlap_score
            + 0.15 * empathy_score
            + pivot_bonus
            - repetition_penalty,
            0.0,
            1.0,
        )

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
        features = features if features is not None else self.last_features.get(topic_idx)
        if features is None:
            features = self._build_feature_map("", {}, time.time())[topic_idx]

        clipped_reward = float(np.clip(reward, 0.0, 1.0))
        self.strategy.update(topic_idx, clipped_reward, features)
        self.topic_counts[topic_idx] += 1
        self.topic_reward_sums[topic_idx] += clipped_reward
        self.topic_last_timestamp[topic_idx] = timestamp or time.time()
        self.last_features[topic_idx] = features
        self.last_reward_value = clipped_reward

    def add_to_history(self, user_input: str, response: str, topic: str) -> None:
        self.conversation_history.append(
            {
                "user_input": user_input,
                "response": response,
                "topic": topic,
                "timestamp": time.time(),
                "reward": self.last_reward_value,
                "negative_feedback": self.last_negative_feedback,
                "question_count": self._last_question_count,
                "response_length": self._last_response_length,
            }
        )

    def get_topic_stats(self) -> Dict[str, Dict[str, float]]:
        now = time.time()
        feature_map = self._build_feature_map("", {}, now)
        stats: Dict[str, Dict[str, float]] = {}
        for idx, topic in enumerate(self.topics):
            stats[topic] = {
                "count": float(self.topic_counts[idx]),
                "avg_reward": float(
                    self.topic_reward_sums[idx] / self.topic_counts[idx]
                    if self.topic_counts[idx] > 0
                    else 0.0
                ),
                "expected_reward": self.strategy.get_expected_reward(idx, feature_map[idx]),
                "neg_feedback": float(self.topic_negative_score[idx]),
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
        self._apply_decay(now)

        context_lower = context.lower()
        intensity = float(emotion.get("intensity", 0.5))
        confidence = float(emotion.get("confidence", 0.5))
        pivot_pressure = np.clip(self.pending_pivot_pressure, 0.0, 1.0)

        feature_map: Dict[int, np.ndarray] = {}
        for idx, topic in enumerate(self.topics):
            novelty = 1.0 / (1.0 + self.topic_counts[idx])
            recency_feature = self._recency_score(idx, now)
            topic_in_context = 1.0 if topic.lower() in context_lower else 0.0
            negative_signal = float(self.topic_negative_score[idx])

            features = np.array(
                [
                    1.0,
                    intensity,
                    confidence,
                    recency_feature,
                    novelty,
                    topic_in_context,
                    negative_signal,
                    pivot_pressure,
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

    def _apply_decay(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_decay_time)
        if elapsed <= 0.0:
            return
        decay = float(np.exp(-elapsed / self.recency_halflife))
        self.topic_negative_score *= decay
        self.pending_pivot_pressure *= decay
        self.last_negative_feedback *= decay
        self._last_decay_time = now

    def _negativity_score(self, text: str) -> float:
        if not text:
            return 0.0
        normalized = text.lower()
        negative_hits = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text or kw in normalized)
        positive_hits = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text or kw in normalized)
        score = max(0.0, negative_hits - 0.5 * positive_hits)
        return float(np.clip(score / 3.0, 0.0, 1.0))

    def _pivot_intent_score(self, text: str) -> float:
        if not text:
            return 0.0
        matches = sum(1 for kw in self.PIVOT_KEYWORDS if kw in text)
        return float(np.clip(matches / 2.0, 0.0, 1.0))

    def _overlap_score(self, user_input: str, response: str) -> float:
        user_tokens = set(self.TOKEN_PATTERN.findall(user_input.lower()))
        response_tokens = set(self.TOKEN_PATTERN.findall(response.lower()))
        if not user_tokens or not response_tokens:
            return 0.4 if user_tokens or response_tokens else 0.0
        intersection = len(user_tokens & response_tokens)
        union = len(user_tokens | response_tokens)
        return float(np.clip(intersection / union, 0.0, 1.0))
