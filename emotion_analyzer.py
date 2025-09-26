import os
import re
from typing import Dict, List

import openai
from dotenv import load_dotenv


class EmotionAnalyzer:
    def __init__(self, model: str | None = None):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_CHAT_COMPLETION_MODEL", "gpt-3.5-turbo")

        self.base_emotions = [
            "喜び",
            "悲しみ",
            "怒り",
            "驚き",
            "恐れ",
            "嫌悪",
            "期待",
            "信頼",
        ]

        self.intensity_levels = ["弱い", "中程度", "強い"]

        self.emotion_combinations = [
            "喜びと期待",
            "悲しみと恐れ",
            "怒りと嫌悪",
            "驚きと喜び",
            "信頼と期待",
        ]

    def analyze_emotion(self, text: str) -> Dict:
        """LLM を用いてテキストの感情を推定する。"""
        if not text.strip():
            return self._get_default_emotion()

        prompt = f"""
        以下の発話から感情を分析してください。

        発話: {text}

        次の形式で結果を出力してください。
        1. 主要な感情: (最大 3 つ、読点区切り)
        2. 感情の強度: (0.0-1.0)
        3. 感情の組み合わせ: (該当しない場合は「なし」)
        4. 感情の変化: (例: 上昇/低下/安定)
        5. 感情の要因: (短く理由を説明)
        6. 信頼度: (0.0-1.0)
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは日本語の感情分析を行う専門家です。指定フォーマットで簡潔に回答してください。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            analysis = response.choices[0].message.content
            return self._parse_analysis(analysis)
        except Exception as exc:  # noqa: BLE001 - keep broad to ensure graceful fallback
            print(f"感情分析でエラーが発生しました: {exc}")
            return self._get_default_emotion()

    def _parse_analysis(self, analysis: str) -> Dict:
        result = {
            "primary_emotions": [],
            "intensity": 0.5,
            "emotion_combination": "なし",
            "emotion_change": "不明",
            "reason": "",
            "confidence": 0.5,
        }

        if not analysis:
            return result

        for line in analysis.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            value = self._extract_value(line)

            if "主要な感情" in line or re.match(r"^\s*1[\.．・)]", line):
                result["primary_emotions"] = self._split_list(value)
            elif "感情の強度" in line or re.match(r"^\s*2[\.．・)]", line):
                result["intensity"] = self._extract_float(value, default=0.5)
            elif "感情の組み合わせ" in line or re.match(r"^\s*3[\.．・)]", line):
                result["emotion_combination"] = value or "なし"
            elif "感情の変化" in line or re.match(r"^\s*4[\.．・)]", line):
                result["emotion_change"] = value or "不明"
            elif "感情の要因" in line or re.match(r"^\s*5[\.．・)]", line):
                result["reason"] = value
            elif "信頼度" in line or re.match(r"^\s*6[\.．・)]", line):
                result["confidence"] = self._extract_float(value, default=0.5)

        if not result["primary_emotions"]:
            result["primary_emotions"] = ["中立"]

        result["intensity"] = max(0.0, min(1.0, result["intensity"]))
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))

        return result

    def _extract_value(self, line: str) -> str:
        cleaned = re.sub(r"^\s*\d+\s*[\.．・\)]?", "", line).strip()
        if "：" in cleaned:
            _, _, remainder = cleaned.partition("：")
        elif ":" in cleaned:
            _, _, remainder = cleaned.partition(":")
        else:
            remainder = cleaned
        return remainder.strip()

    def _split_list(self, value: str) -> List[str]:
        if not value:
            return []
        parts = re.split(r"[、,]\s*", value)
        return [part for part in (p.strip() for p in parts) if part]

    def _extract_float(self, value: str, default: float) -> float:
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return default
        try:
            return float(match.group())
        except ValueError:
            return default

    def get_emotion_expression(self, emotion_data: Dict) -> str:
        prompt = f"""
        以下の感情情報をもとに、VTuber の演技指針を短く提案してください。

        主要な感情: {', '.join(emotion_data.get('primary_emotions', []))}
        感情の強度: {emotion_data.get('intensity', 0.5)}
        感情の組み合わせ: {emotion_data.get('emotion_combination', 'なし')}
        感情の変化: {emotion_data.get('emotion_change', '不明')}

        次の形式で日本語で回答してください。
        1. 表情
        2. 声の調子
        3. 体の動き
        4. 感情表現の強さ (0.0-1.0)
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは VTuber の演技指導を行う専門家です。指示は簡潔にしてください。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            print(f"感情表現の生成でエラーが発生しました: {exc}")
            return "通常の表情"

    def get_emotion_history(self, text_history: List[str]) -> List[Dict]:
        emotions: List[Dict] = []
        for text in text_history:
            emotions.append(self.analyze_emotion(text))
        return emotions

    def _get_default_emotion(self) -> Dict:
        return {
            "primary_emotions": ["中立"],
            "intensity": 0.5,
            "emotion_combination": "なし",
            "emotion_change": "不明",
            "reason": "感情分析を実行できませんでした。",
            "confidence": 0.0,
        }
