import argparse
import glob
import os
from PIL import Image


def main(args):
    # scale_list = [0.75, 0.6, 1 / 3]
    scale_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    shortest_edge = args.shortest_edge

    for dataset in args.input:
        path_list = sorted(glob.glob(os.path.join(dataset, '*')))
        save_path = os.path.join(os.path.dirname(dataset), f"{os.path.basename(dataset)}_multiscaleHR_shortest{args.shortest_edge}")
        os.makedirs(save_path, exist_ok=True)
        
        for path in path_list:
            print(path)
            basename = os.path.splitext(os.path.basename(path))[0]

            img = Image.open(path)
            width, height = img.size
            width_original, height_original = img.size
            for idx, scale in enumerate(scale_list):
                # print(f'\t{scale:.2f}')

                if min(int(width * scale), int(height * scale)) >= shortest_edge:

                    rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
                    rlt.save(os.path.join(save_path, f'{basename}T{idx}.png'))

            # save the smallest image which the shortest edge is 512
            if width_original < height_original :
                ratio = height_original / width_original
                width = shortest_edge
                height = int(width * ratio)
            else:
                ratio = width_original / height_original
                height = shortest_edge
                width = int(height * ratio)

            assert min(height, width) >= shortest_edge, f" The width-height {width}-{height} is not suitable."

            rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
            rlt.save(os.path.join(save_path, f'{basename}S.png'))

            if min(width_original, height_original) >= shortest_edge:
                os.system(f'cp -r {path} {save_path}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=list, default=['/home/notebook/data/group/chendu/dataset/DIV2K/trainHR',
                                                        '/home/notebook/data/group/chendu/dataset/Flickr2K/trainHR'],
                        help='Input folder')
    parser.add_argument('--shortest_edge', type = int, default=512)
    args = parser.parse_args()
    main(args)
