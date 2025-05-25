# RecoMate Core

RecoMateのコア処理を担当するPythonモジュール群。音声認識、感情分析、自然な会話生成、音声合成、表情制御の中核となる処理を提供します。

## 概要

このリポジトリは、RecoMateのバックエンド処理を担当するPythonモジュール群です。主に以下の機能を提供します：

- 音声認識と応答生成
- 感情分析と表情制御
- VOICEVOXを使用した音声合成
- トピックバンディットによる会話の最適化

## 技術仕様

### 必要条件

- Python 3.8以上
- OpenAI APIキー
- VOICEVOX（音声合成エンジン）
- マイク（音声入力用）

### 依存パッケージ

主要な依存パッケージ：
- OpenAI API - 自然言語処理
- Pygame - 音声再生
- SpeechRecognition - 音声認識
- NumPy - 数値計算
- SoundDevice - オーディオ処理

## インストール方法

1. リポジトリをクローン
```bash
git clone https://github.com/yourusername/recomate-core.git
cd recomate-core
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

## モジュール構成

- `main.py`: メインプログラム
- `text_to_speech.py`: 音声合成処理
- `emotion_analyzer.py`: 感情分析
- `topic_bandit.py`: トピック選択アルゴリズム
- `vtuber_model.py`: 表情制御

## 開発者向け情報

### テスト実行

```bash
python main.py
```

### デバッグモード

環境変数に以下を追加：
```
DEBUG=true
```

## クレジット

### VOICEVOX
- [VOICEVOX](https://voicevox.hiroshiba.jp/) - 音声合成エンジン
- 利用規約: [VOICEVOX利用規約](https://voicevox.hiroshiba.jp/term/)
- ライセンス: [VOICEVOXライセンス](https://github.com/VOICEVOX/voicevox/blob/master/LICENSE)

## ライセンス

MIT License

## 注意事項

- このモジュール群はRecoMateのコア処理を担当する内部コンポーネントです
- 直接の使用は想定されていません
- 商用利用の場合は、各ライブラリのライセンスを確認してください
- VOICEVOXの利用については、[VOICEVOX利用規約](https://voicevox.hiroshiba.jp/term/)に従ってください

## 関連リポジトリ

- [RecoMate](https://github.com/yourusername/recomate) - メインアプリケーション 