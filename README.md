YOLO X + ByteTrack による**複数人同時プレイ**に対応した機械学習版「だるまさんがころんだ」です。
オープンキャンパスの出し物として作りました。

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

## 手法

### ByteTrack

ByteTrackは、2021年10月に公開されたオブジェクトトラッキングのモデルです。今回は、撃たれた人を追跡するための仕組みとして使っています。Deep SortのようにReIDは使用していません。カルマンフィルタによるバウンディングボックスの移動予測のみでフレームを紐づけています。最初は高い確度のバウンディングボックスに対して、次に低い確度のバウンディングボックスに対して紐付けを行うことで、隠れたオブジェクトでも不検知になりづらい仕組みになっています。

<p align="center"><img src="assets/sota.png" width="500"/></p>

> [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](https://arxiv.org/abs/2110.06864)
> 
> Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang
> 
> *[arXiv 2110.06864](https://arxiv.org/abs/2110.06864)*

### YOLO X

YOLO Xは、2021年8月に公開されたアンカーフリーの物体検出モデルです。今回は、YOLO Xで検出したバウンディングボックス内をフレーム間差分で差分を算出し動体検知しています。

### 動体検知

フレーム間差分を使っています。ただし、単純に前の差分と比較するだけでは些細な自然光でも大きく変化してしまうことがあります。そこで、移動平均(蓄積されたフレーム,最新フレーム間の差分)で変化を和らげています。
MoveNetといった体のランドマークをとる方法でも検知は可能ですが、フレーム毎に推論にブレが生じたりターン毎に距離が違ったりするため、難しいです。

## 雰囲気

普通の「だるまさんがころんだ」とイカゲーム版の「だるまさんがころんだ」の両方を用意しています。

### ノーマル版
![IMG_0663](https://user-images.githubusercontent.com/33394165/153808593-d8490b3c-9183-42ca-959e-4aa0407ed9f8.png)
### イカゲーム版
![IMG_0664](https://user-images.githubusercontent.com/33394165/153808678-3c4cf008-b73a-4786-8a00-20942a5eaed9.png)

### チュートリアル
![IMG_0665](https://user-images.githubusercontent.com/33394165/153808695-4f3e3373-e801-4c86-9e7f-56f5dff3304b.png)
約5Mの位置に赤い線を引いてます！
![IMG_0666](https://user-images.githubusercontent.com/33394165/153808717-4b0a0062-97ff-4316-af08-55bb35ea54d2.png)
![IMG_0667](https://user-images.githubusercontent.com/33394165/153808738-6e4d9c69-ab75-4758-b38e-a80055d0a677.png)
![IMG_0668](https://user-images.githubusercontent.com/33394165/153808858-1dd6b532-c792-4490-bbcd-0da4aa8a8b76.png)
### カウントダウン
![IMG_0669](https://user-images.githubusercontent.com/33394165/153808890-78fdf278-6fea-4250-ab1c-7630d0b18690.png)
![IMG_0671](https://user-images.githubusercontent.com/33394165/153808920-42cfca9a-5098-4a4f-9ca8-d6d4211987a8.png)
### 「だるまさんがころんだ」
イカゲームの原作の韓国語では、「ムグンファ コッチ ピオッスムニダ」
![IMG_0672](https://user-images.githubusercontent.com/33394165/153808937-cec7e599-355c-4ecc-8562-b34ae0dd2778.png)
### ストップ！
![IMG_0673](https://user-images.githubusercontent.com/33394165/153808957-0fa3ae3d-0436-4358-bc5b-c9dab16f1ccb.png)
### 止まれなかった...
撃たれた！
![IMG_0674](https://user-images.githubusercontent.com/33394165/153808975-e2759999-e5df-4f6f-b7a6-46bff1908d23.png)
### アウトになっちゃった...
人がいない時、もしくは全員が撃たれた時はこのリザルト画面に移行します。
制限ターン内にキーボードのGを押すと、ゲームクリアです。
![IMG_0675](https://user-images.githubusercontent.com/33394165/153808996-32842dd0-dbee-421d-90d0-b11ab1d1413b.png)

### 終了！リトライする？
![IMG_0676](https://user-images.githubusercontent.com/33394165/153809018-3bbda634-775f-4a6b-bdab-c81140472e57.png)
