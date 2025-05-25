# AI VTuber

AIを搭載したVTuberシステム。音声認識、感情分析、自然な会話生成、音声合成、表情制御を組み合わせて、インタラクティブなVTuber体験を提供します。

## 主な機能

- 音声認識による自然な対話
- 感情分析による適切な応答生成
- VOICEVOXを使用した高品質な音声合成
- 感情に応じた表情の切り替え
- トピックバンディットによる会話の最適化

## 必要条件

- Python 3.8以上
- OpenAI APIキー
- VOICEVOX（音声合成エンジン）
- マイク（音声入力用）

## インストール方法

1. リポジトリをクローン
```bash
git clone https://github.com/yourusername/ai-vtuber.git
cd ai-vtuber
```

2. 依存関係をインストール
```bash
pip install -r requirements.txt
```

3. 環境変数の設定
`.env`ファイルを作成し、以下の内容を設定：
```
OPENAI_API_KEY=your_api_key_here
```

4. VOICEVOXのインストールと起動
- [VOICEVOX公式サイト](https://voicevox.hiroshiba.jp/)からダウンロード
- インストール後、VOICEVOXを起動

## 使用方法

1. VOICEVOXを起動
2. プログラムを実行
```bash
python main.py
```
3. マイクに向かって話しかける

## プロジェクト構成

- `main.py`: メインプログラム
- `text_to_speech.py`: 音声合成処理
- `emotion_analyzer.py`: 感情分析
- `topic_bandit.py`: トピック選択アルゴリズム
- `vtuber_model.py`: 表情制御

## クレジット

### VOICEVOX
- [VOICEVOX](https://voicevox.hiroshiba.jp/) - 音声合成エンジン
- 利用規約: [VOICEVOX利用規約](https://voicevox.hiroshiba.jp/term/)
- ライセンス: [VOICEVOXライセンス](https://github.com/VOICEVOX/voicevox/blob/master/LICENSE)

### その他のライブラリ
- OpenAI API - 自然言語処理
- Pygame - 音声再生
- SpeechRecognition - 音声認識
- NumPy - 数値計算
- SoundDevice - オーディオ処理

## ライセンス

MIT License

## 注意事項

- このプロジェクトは研究・教育目的で作成されています
- 商用利用の場合は、各ライブラリのライセンスを確認してください
- VOICEVOXの利用については、[VOICEVOX利用規約](https://voicevox.hiroshiba.jp/term/)に従ってください

## キャラクター画像について

本リポジトリのキャラクター画像は、OpenAIのChatGPT（DALL·E）を利用して生成したものです。 