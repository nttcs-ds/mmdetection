# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import numpy
from os.path import basename, isdir
from os import chmod, makedirs
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector


def show_result(model,
                img,
                result,
                out,
                score_thr=0.3,
                title='result',
                wait_time=0):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        out_file=out)


def create_batch(images, batch_size):
    chunked_list = []
    for i in range(0, len(images), batch_size):
        chunked_list.append(images[i: i + batch_size])
    return chunked_list


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img_output_dir', help='Output directory')
    parser.add_argument('params_output_dir', help='Output directory')
    parser.add_argument('img', nargs="+", help='Image file')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    img_data = create_batch(args.img, args.batch)
    states_result = []
    labels_result = []
    bboxes_result = []
    filenames = []
    if not isdir(args.img_output_dir):
        makedirs(args.img_output_dir)
        chmod(args.img_output_dir, 0o755)
    for d in tqdm(img_data):
        result, intermediate = inference_detector(model, d)
        states, labels, bboxes = intermediate
        filenames.extend(d)
        states_result.append(states.to('cpu').detach().numpy().copy())
        labels_result.append(labels.to('cpu').detach().numpy().copy())
        bboxes_result.append(bboxes.to('cpu').detach().numpy().copy())
        for f, r in zip(d, result):
            out_file = args.img_output_dir + basename(f)
            show_result(model, f, r,
                        score_thr=args.score_thr, out=out_file)
    result_dir = args.params_output_dir
    if not isdir(result_dir):
        makedirs(result_dir)
        chmod(result_dir, 0o755)
    with open(result_dir + "states.npz", "wb") as f:
        numpy.savez(f, numpy.vstack(states_result))
    with open(result_dir + "labels.npz", "wb") as f:
        numpy.savez(f, numpy.vstack(labels_result))
    with open(result_dir + "bboxes.npz", "wb") as f:
        numpy.savez(f, numpy.vstack(bboxes_result))
    with open(result_dir + "filenames.txt", "w") as f:
        for fname in filenames:
            f.write(basename(fname) + "\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)
