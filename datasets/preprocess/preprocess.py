import os
import sys
import shutil
import argparse
import tarfile
import zipfile


def mini_imagenet():
    with tarfile.open('mini_imagenet_full_size.tar.bz2', 'r') as tar:
        tar.extractall()
    os.rename('mini_imagenet_full_size', 'mini_imagenet')


def tiered_imagenet():
    with tarfile.open('tiered_imagenet.tar', 'r') as tar:
        tar.extractall()


def CIFAR_FS():
    phase_list = ['train', 'val', 'test']
    with zipfile.ZipFile('cifar100.zip', 'r') as zip_ref:
        zip_ref.extractall()
    for phase in phase_list:
        os.makedirs('CIFAR_FS/{}'.format(phase))
    for phase in phase_list:
        classes_info_dir = 'cifar100/splits/bertinetto/{}.txt'.format(phase)
        f = open(classes_info_dir, 'r')
        for line in f.readlines():
            class_name = line.strip()
            shutil.move('cifar100/data/{}'.format(class_name), 'CIFAR_FS/{}/{}'.format(phase, class_name))
    shutil.rmtree('cifar100')


def CUB():
    phase_list = ['train', 'val', 'test']
    with tarfile.open('CUB_200_2011.tgz', 'r') as tar:
        tar.extractall()
    for phase in phase_list:
        os.makedirs('CUB/{}'.format(phase))
    for phase in phase_list:
        classes_info_dir = 'preprocess/CUB_split_{}.txt'.format(phase)
        f = open(classes_info_dir, 'r')
        for line in f.readlines():
            class_name = line.strip()
            shutil.move('CUB_200_2011/images/{}'.format(class_name), 'CUB/{}/{}'.format(phase, class_name))
    os.remove('attributes.txt')
    shutil.rmtree('CUB_200_2011')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', choices=['mini_imagenet, tiered_imagenet, CIFAR_FS, CUB'], 
                        help='Dataset name to preprocess.')
    args = parser.parse_args()

    for dataset in args.datasets:
        if os.path.isdir(dataset):
            shutil.rmtree(dataset)
        getattr(sys.modules[__name__], dataset)()
