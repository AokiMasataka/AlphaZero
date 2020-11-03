#Alpha Zero

###動作環境

windows10
<br>
Anaconda
<br>
python 3.7
<br>
pytorch 1.6.0
<br>
tqdm 4.48.2
<br>
numpy 1.19.1

##モデルを学習させる

お好みのディレクトリにクローンします
```
git clone https://github.com/AokiMasataka/AlphaZero.git
```
クローンしたディレクトリで、モデルを最初から学習させる場合は以下のコマンドを実行します。
```
python train.py
```
学習には時間がかかる為、途中からの学習したい場合は
<br>
前回学習し終えたmodelGenを付け足してください。(modelGenは数字のみ)
```
python train.py modelGen
```
##モデルと対戦する

学習し終えたmodelと対戦するには play.py を実行します。
modelGenには対戦したい、modelの世代の数字を入れます。
```
python play.py modelGen
```
##config.pyについて

このファイルはモデルや探索の深さ、などのハイパーパラメータを操作することができます。
<br>
学習が遅いと感じたり、メモリサイズになどを考慮して調整してください。