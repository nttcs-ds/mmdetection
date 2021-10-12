# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img', nargs="+", help='Image file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result, intermediate = inference_detector(model, args.img)
    # states = [batch, query, dim]
    # labels = [batch, query, cls]
    # bboxes = [batch, query, 4]
    states, labels, bboxes = intermediate
    print(f"states.shape:{states.shape}")
    print(f"labels.shape:{labels.shape}")
    print(f"bboxes.shape:{bboxes.shape}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
