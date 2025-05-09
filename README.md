# AI VTuber

AIを活用した自律型VTuberシステム。音声認識、感情分析、自然な会話生成、表情制御を統合したシステムです。

## 機能

- 🎤 音声認識と応答生成
- 😊 感情分析と表情制御
- 🎯 トピック選択と管理
- 🎨 3Dモデル制御とアニメーション

## 必要条件

- Python 3.8以上
- OpenAI APIキー
- 必要なPythonパッケージ（requirements.txtに記載）

## インストール

1. リポジトリをクローン
```bash
git clone https://github.com/yourusername/ai-vtuber.git
cd ai-vtuber
```

2. 必要なパッケージをインストール
```bash
pip install -r requirements.txt
```

3. 環境変数の設定
`.env`ファイルを作成し、以下の内容を追加：
```
OPENAI_API_KEY=your_api_key_here
```

## 使用方法

1. プログラムを実行
```bash
python main.py
```

2. マイクに向かって話しかける
3. VTuberが応答し、表情を変化させながら会話を続けます

## プロジェクト構成

```
ai-vtuber/
├── main.py              # メインプログラム
├── vtuber_model.py      # 3Dモデル制御
├── emotion_analyzer.py  # 感情分析
├── topic_bandit.py      # トピック選択
├── text_to_speech.py    # 音声合成
├── requirements.txt     # 依存パッケージ
└── README.md           # プロジェクト説明
```

## ライセンス

MITライセンス

## 貢献

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 注意事項

- OpenAI APIの利用には料金が発生します
- 音声認識にはインターネット接続が必要です
- キャラクター画像は別途用意する必要があります

## キャラクター画像について

本リポジトリのキャラクター画像は、OpenAIのChatGPT（DALL·E）を利用して生成したものです。 