# FARO_for_Large_tomato
大玉トマト収穫機の一連の動かし方をまとめる。

## Setup
### shiotani_tomato_ros_system(画像処理)
画像処理はDocker上で実行する。最新のソフトへチェックアウト後、Dcokerイメージを合計四つ作成する。

1.Instance Segmentation

aliasが設定されている場合は`build_instseg` でイメージ作成できる。aliasが設定されていない場合は以下のコマンドにて実行する。

```
cd /home/denso/catkin_ws/src/shiotani_tomato_ros_system/instance_segmentation && docker build -t instance_segmentation -f Dockerfile .
```

cuda10.1を使用するので、`~/catkin_ws/src/instance_segmentation/`下にcuda10.1を配置しておくこと。以下にモデルとcuda10.1が保存されている。
アクセス権が必要なので、アクセスしたい場合は石川まで連絡する。

- モデル、cuda保存先：[shitotani_drive](https://drive.google.com/drive/folders/1St_zGmSexP-WBKQZmUKT2vBYf9QTvqIE?usp=sharing)

2.Stereo Matching

aliasが設定されている場合は`build_sm` でイメージ作成できる。aliasが設定されていない場合は以下のコマンドにて実行する。

```
cd /home/denso/catkin_ws/src/shiotani_tomato_ros_system/stereo_matching && docker build -t stereo_matching -f Dockerfile .
```

cuda10.2を使用するので、`~/catkin_ws/src/stereo_matching/`下にcuda10.2を配置しておくこと。以下にモデルとcuda10.2が保存されている。
アクセス権が必要なので、アクセスしたい場合は石川まで連絡する。

- モデル、cuda保存先：[shitotani_drive](https://drive.google.com/drive/folders/1St_zGmSexP-WBKQZmUKT2vBYf9QTvqIE?usp=sharing)

3. Synthesis

aliasが設定されている場合は`build_synthesis` でイメージ作成できる。aliasが設定されていない場合は以下のコマンドにて実行する。

```
cd /home/denso/catkin_ws/src/shiotani_tomato_ros_system/synthesis && docker build -t synthesis -f Dockerfile .
```

4. Zed mini

aliasが設定されている場合は`build_zedmini` でイメージ作成できる。aliasが設定されていない場合は以下のコマンドにて実行する。

```
cd /home/denso/catkin_ws/src/shiotani_tomato_ros_system/zed_mini && docker build -t zed_mini -f Dockerfile .
```

以下で画像処理ノードをDocker上で起動する。

- docker-compose up

### d_robot(ロボット制御)

[README.md](https://github.com/denso-robot-fa/d_robot/blob/master/README.md)に従ってクローンビルドする。
大玉トマト用の起動コマンドは以下。

```
roslaunch large_tomato large_tomato.launch ip_address:=192.168.250.10
```

## Usage
### detect pedicel

bagファイルを用いた検出を実行する場合、`docker-compose.yaml`内の`zed_mini`ノードをコメントアウトする。
以下のコマンドを各ターミナルにて実行する。

```
docker-compose up
rosbag play -l <zed_m.bag>
rqt_image_view
detect
``` 
`rqt_image_view`するターミナルは`ssh denso@192.168.250.10 -X`としないと手元のPCで表示できないので気を付ける。

### Check approach motion

bagファイルを用いた検出を実行する場合、`docker-compose.yaml`内の`zed_mini`ノードをコメントアウトする。
以下のコマンドを各ターミナルにて実行する。

```
docker-compose up
roslaunch large_tomato large_tomato.launch ip_address:=192.168.250.10
rosbag play -l <zed_m.bag>
``` 
その後対話型コマンドプロンプトを起動し、実行コマンドを入力する。
```
roscd large_tomato/scripts
ipython -i main.py -- __ns:=d_robot
>> harvester.action_for_one_fruit()
```

## Tips
- ローカルのlarge_tomato_log内にログは保存されており、検出した際にログの自動保存がされる。

- 初期位置でアプローチできるbagファイル
`20220128_150406zedm.bag`

- moveitのTFの`orientation`の許容誤差設定パラメータ（仮）
`rosparam set /test_joint_tolerance 0.01`

## Link
- 画像処理

[インスタンス・セグメンテーションの精度・汎化改善](https://appl.dndev.net/sn03/wiki/pages/viewpage.action?pageId=431484814)

[最新モデル：データオーグメンテーション](https://appl.dndev.net/sn03/wiki/pages/viewpage.action?pageId=431992839)

[参考資料：20211020_MTGスライド](https://appl.dndev.net/sn03/wiki/pages/viewpage.action?pageId=412279185)

[shiotani_tomato_ros_system.github](https://github.com/denso-robot-fa/shiotani_tomato_ros_system)

- ロボット制御

[moveit許容誤差：20220207_大玉トマト収穫動作の改善（1）](https://appl.dndev.net/sn03/wiki/pages/viewpage.action?pageId=436045212)

[20211208_大玉トマトテスト事前準備](https://appl.dndev.net/sn03/wiki/pages/viewpage.action?pageId=423959718)

[20220124_大玉ロボット事前動作確認](https://appl.dndev.net/sn03/wiki/pages/viewpage.action?pageId=431997979)

- IFbox関連

[20211223_Inference boxのBIOS更新](https://appl.dndev.net/sn03/wiki/pages/viewpage.action?pageId=426885822)

[20211021_Inference_box_CUDA_challenge2](https://appl.dndev.net/sn03/wiki/display/RBAGR/20211021_Inference_box_CUDA_challenge2)

- Docker

[20211012_shiotani_tomato_ros_systemをdocker-compose upする](https://appl.dndev.net/sn03/wiki/pages/viewpage.action?pageId=409885442)

[20211101_インファレンスボックスでdocker-compose&dockerとホストのROS通信](https://appl.dndev.net/sn03/wiki/pages/viewpage.action?pageId=414115051)

- モデルパス

[shitotani_drive](https://drive.google.com/drive/folders/1St_zGmSexP-WBKQZmUKT2vBYf9QTvqIE?usp=sharing)

- ログデータ

[04_大玉トマトログデータ](https://appl.dndev.net/sn03/wiki/pages/viewpage.action?pageId=415875458&src=contextnavpagetreemode)

