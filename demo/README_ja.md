# 中間データ取得

`mmdet.apis.inference_detector` の返り値に以下の３つを追加しました。

- feature embedding(256次元)
- object keyに対する分類の結果
- object keyに対するbounding box

`inference_detector` の返り値を以下のようにして受け取ると、 `intermediate` に上記３つの情報が入っています。第一返り値の `result` の方は `inference_detector` の従来の返り値です。

```
result, intermediate = inference_detector(model, args.img)
```

## feature embedding
Deformable DETRの推論における最終レイヤーのhidden embeddingです。PytorchのTensor型で[batch size, number of object query, dimension of the hidden layer] の大きさになっています。

## object keyに対する分類の結果
Deformable DETRの推論における分類の最終結果です。PytorchのTensor型で[batch size, number of object query, number of classes]の大きさになっています。
Softmax 適用前の結果なので確率値にはなっていません。

## object keyに対するbounding box
Deofrmable DETRの推論におけるbounding boxの値です。PytorchのTensor型で[batch size, number of object query, 4]の大きさとなっています。それぞれの値は画像の縦横における割合となっています。pixelによる位置に変換するにはそれぞれ画像のpixel数をかけてやる必要があります。

# サンプルプログラム
`intermediate_demo.py` は中間データ取得し、それぞれのshapeを表示するプログラムです。

- `config` : モデルの設定ファイル。現在は `configs/deformable_detr/deformable_detr_twostage_refine_x101_32x4d_fpn_dconv_c3-c5_visual_genome.py` にしか対応していません。
- `checkpoint` : 学習済みモデルのファイル。
- `img` : 画像ファイルを指定します。複数指定できます。
- `--device` : 推論に用いるデバイスを指定するオプションです。デフォルトは `cuda:0` です。CPUのみで推論する場合は `cpu` を指定します。

実行例
```
python demo/intermediate_demo.py configs/deformable_detr/deformable_detr_twostage_refine_x101_32x4d_fpn_dconv_c3-c5_visual_genome.py /path/to/model  /path/to/image

states.shape:torch.Size([13, 300, 256])
labels.shape:torch.Size([13, 300, 1600])
bboxes.shape:torch.Size([13, 300, 4])

```