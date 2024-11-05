# Wavelet Scatteringを用いた深層生成モデル
# 概要
本レポジトリでは、Wavelet Scatteringを用いた深層生成モデル([Angles & Mallat, 2018](https://arxiv.org/pdf/1805.06621.pdf))とその改良モデルのライブラリを
公開しています。
Wavelet Scatteringを用いた深層生成モデルは、VAEのようにエンコーダーとデコーダーで構成される画像生成モデルです。
エンコーダーが特徴的で、Wavelet ScatteringとPCAで構成されています。
そのため、解釈性が高く、BackPropagation(BP) による学習が不要です。
ただし、乱数をデコーダーに入力して得られる生成画像は、大きな割合で不自然な画像になります。
そこで、エンコーダーの出力（潜在変数ベクトル）にガウス確率変数ベクトルを加えてデコーダーを学習するようモデルを改良しました。
これにより，汎化が改善され，自然な画像の生成が可能となりました。

本レポジトリのコードで、既存のWavelet Scatteringを用いた深層生成モデルと改良モデルの学習と推論（画像生成、画像復元）を実行できます。
なお、既存モデルの解説と実験結果をブログ([Wavelet Scatteringを用いた深層生成モデルとその実験結果の解説](https://zenn.dev/tkmtark/articles/c4009e0a5030dc))にまとめています。
よろしければご覧ください。改良モデルの解説と実験結果はIBIS2024で発表します。

# 使用法
画像データを学習用、ヴァリデーション用、テスト用等に分けて別々のフォルダに設置します（画像は正方形を想定しています）。
Configファイルにデータを格納したフォルダのパスや画像サイズなど各種設定項目を記入します。
Configファイルは、例```config_example.yml```を参考にして作成してください。
コメントに各設定項目の説明を記載しています。
## 学習
以下のコマンドで既存モデルまたは改良モデルを学習できます。
```shell
python <tain_sgm.py or tain_ssgm.py> <config file path>
```
既存モデルの学習には```tain_sgm.py```、改良モデルの学習には```tain_ssgm.py```を使用します。
オプション引数を設定することで、実験管理ツールのMlflowとTensorboardどちらを使うか、エンコードをスキップするか、SEEDを固定するか、を選択できます。

例：
既存モデルを学習する場合
```shell
python tain_sgm.py ./configs/config_celeba_sgm.yml
```

以下のコマンドで、指定した実験管理ツール（MlflowまたはTensorboard）を起動して、学習時の訓練データにおけるロスと
ヴァリデーションデータにおけるロスの履歴を確認できます。

Mlflow UIの起動
```shell
mlflow ui --backend-store-uri <log directory path>
```

Tensorboardの起動
```shell
tensorboard --logdir <log directory path>
```
```<log directory path>```はConfigファイルに設定したパスを指定してください。
## 推論
テストデータと復元用データをConfigファイルで指定したフォルダに格納します。
以下のコマンドで、学習した既存モデルまたは改良モデルを使って画像生成、画像復元、テストロスの算出を実行できます（画像生成には標準正規乱数を使用）。
```shell
python infer.py <"sgm" or "ssgm"> <config file path> <epoch>
```
第一引数は、既存モデルで推論する場合は```sgm```、改良モデルで推論する場合は```ssgm```と指定します。
第三引数```<epoch>```で指定したエポック時の学習モデルで推論します。

例：
100エポックまで学習した改良モデルで推論する場合
```shell
 python infer.py "ssgm" ./configs/config_celeba_ssgm.yml 100
```

推論結果は、上記の学習時のロスの確認と同様にMlflow UIを起動して確認できます。

# 動作確認時のバージョン
Pythonと主なパッケージの動作確認時のバージョンは以下の通りです。
```
Python                      3.10.13 

Packages
- torch                     2.1.0+cu118
- torchvision               0.16.0+cu118
- kymatio                   0.3.0
- scikit-learn              1.3.2
- numpy                     1.24.1
- tensorboard               2.15.1
- mlflow                    2.11.3
```

# References
- Angles & Mallat (2018) [generative networks as inverse problems with scattering transforms](https://arxiv.org/pdf/1805.06621.pdf), ICLR.
   ([著者のコード](https://github.com/tomas-angles/generative-scattering-networks))
- [Wavelet Scatteringを用いた深層生成モデルとその実験結果の解説](https://zenn.dev/tkmtark/articles/c4009e0a5030dc)
