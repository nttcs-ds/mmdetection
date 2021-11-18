# 中間データ取得

`mmdet.apis.inference_detector` の返り値に以下の３つを追加しました。

- feature embedding(256次元)
- object keyに対する分類の結果
- object keyに対するbounding box
- object keyに対するattributeの分類の結果

`inference_detector` の返り値を以下のようにして受け取ると、 `intermediate` に上記4つの情報が入っています。第一返り値の `result` の方は `inference_detector` の従来の返り値です。

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

## object keyに対するattribute分類の結果
DeformableDETRの推論におけるattribute分類の最終結果です。PytorchのTensor型で[batch size, number of object query, number of attributes + 1]の大きさになっています。attributeは0番から始まり、最後は属性なしに対応しています。
Softmax適用前の結果なので確率値にはなっていません。

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
attr.shape:torch.Size([13, 300, 401]))
```

`coco_inference.py` はMS COCOのデータに対して推論を行い、intermediateで与えられるembeddingをファイルに保存します。embeddingのサイズが大きくなるため、データを16分割して実行します。1度の実行では `chunk_id` で指定したshardに対してのみ実行されないので、全データに対して行う際には `chunk_id` を0～15に変更して実行してください。

- `config` : モデルの設定ファイル。現在は `configs/deformable_detr/deformable_detr_twostage_refine_x101_32x4d_fpn_dconv_c3-c5_visual_genome.py` にしか対応していません。
- `checkpoint` : 学習済みモデルのファイル。
- `annotation` : MS COCOのアノテーションファイル(instance_train2017.json等)
- `image_dir`: MS COCOの画像ファイルが存在するディレクトリへのパス
- `output_dir` : 生成したembeddingを出力するディレクトリ。
- `chunk_id` : 16分割した中でどのshardを用いるかを指定するID。0～15の範囲で指定する。
- `--device` : 推論に用いるデバイスを指定するオプションです。デフォルトは `cuda:0` です。CPUのみで推論する場合は `cpu` を指定します。

プログラムが正常に終了したあと、 `output_dir` に指定したディレクトリに以下のファイルが生成されます。npz形式のファイルを読み込む際のkeyは `arr_0` になります。

- `states.npz` : feature emgeddingをnpz形式で保存したファイル
- `labels.npz` : object keyに対する分類の結果をnpz形式で保存したファイル
- `bboxes.npz` : object keyに対するbounding boxをnpz形式で保存したファイル
- `attr.npz` : object keyに対するattribute分類の結果をnpz形式で保存したファイル
- `filenames.txt` : 画像ファイル名のリスト
```
