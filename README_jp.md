# rock-paper-scissors-rllib

## 概要
- ray.rllibで実装した初学者向けの強化学習のサンプルソース

## 内容
- .devcontainer
    - 開発環境をdockerコンテナで提供するための設定一式
- train.ipynb
    - じゃんけんシミュレータを例にとって実装した強化学習のサンプルソース
    - jupyter notebook形式で強化学習の流れを説明

## 環境構築手順
1. VSCodeをインストール
1. 拡張機能devcontainerをインストール
1. dockerをインストール
1. Ctrl+Shift+Pを押下してbuild containerを選択
    - docker-desktopをインストールした場合はwslを有効にすること
    - VSCodeがwslを起動できない場合以下のコマンドを実行してOSを再起動すると良い
```powershell
sudo netsh reset
```

## 実行手順
1. ipykernelをpython:3.10の環境で起動する
1. train.ipynbの各セルを上から実行していく
