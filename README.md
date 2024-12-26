#株価予測モデル
nttの株価データからの株価予測モデルです。使用したモデルはLSTMです。
##概要
nttの株価データ（ｃｓｖ形式）を受け取り予測モデルを作ります。
各データの終値（close）をもとにLSTMモデルを用いて学習し、予測を行います。
目的変数は終値以外にも設定可能です。
LSTMに用いるタイムステップ数を変えることで学習の精度を向上させられます。
評価指標として、RMSEとR-squaredをアウトプットします。
＃インストール
```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn 
pip install tensorflow
pip install datetime
```
##使い方
ローカルにインストール
```bash
リポジトリをクローン
git clone https://github.com/username/repository.git

ディレクトリに移動
cd repository

使用
python demo.py
```
