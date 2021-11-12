# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import numpy
from os.path import basename
from os import chmod, makedirs
from tqdm import tqdm
from mmdet.apis import inference_detector, init_detector
from mmdet.datasets import CocoDataset


CHUNK_NUM = 16


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('annotation', help='Annotation file')
    parser.add_argument('image_dir', help='Coco image file dir')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument(
        'chunk_id', type=int, help='Chunk ID of datasets (from 0 to 15)')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def create_batch(annotations, image_dir, batch_size):
    dataset = CocoDataset(ann_file=annotations,
                          pipeline=[], img_prefix=image_dir)
    anno = dataset.load_annotations(annotations)
    chunked_list = []
    for i in range(0, len(anno), batch_size):
        chunked_list.append([args.image_dir + "/" + f["file_name"]
                             for f in anno[i: i+batch_size]])
    return chunked_list


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # create inputs
    data = create_batch(args.annotation, args.image_dir, args.batch)

    # inference
    states_result = []
    labels_result = []
    bboxes_result = []
    attr_result = []
    filenames = []
    start = len(data) * args.chunk_id // CHUNK_NUM
    end = len(data) * (args.chunk_id + 1) // CHUNK_NUM
    for d in tqdm(data[start: end]):
        result, intermediate = inference_detector(model, d)
        # states = [batch, query, dim]
        # labels = [batch, query, cls]
        # bboxes = [batch, query, 4]
        # attributes = [batch, query, attr]
        states, labels, bboxes, attr = intermediate
        filenames.extend(d)
        states_result.append(states.to('cpu').detach().numpy().copy())
        labels_result.append(labels.to('cpu').detach().numpy().copy())
        bboxes_result.append(bboxes.to('cpu').detach().numpy().copy())
        attr_result.append(attr.to('cpu').detach().numpy().copy())
    result_dir = f"{args.output_dir}/{args.chunk_id}/"
    makedirs(result_dir, True)
    chmod(result_dir, 0o755)
    with open(result_dir + "states.npz", "wb") as f:
        numpy.savez(f, numpy.vstack(states_result))
    with open(result_dir + "labels.npz", "wb") as f:
        numpy.savez(f, numpy.vstack(labels_result))
    with open(result_dir + "bboxes.npz", "wb") as f:
        numpy.savez(f, numpy.vstack(bboxes_result))
    with open(result_dir + "attr.npz", "wb") as f:
        numpy.savez(f, numpy.vstack(attr_result))
    with open(result_dir + "filenames.txt", "w") as f:
        for fname in filenames:
            f.write(basename(fname) + "\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)
