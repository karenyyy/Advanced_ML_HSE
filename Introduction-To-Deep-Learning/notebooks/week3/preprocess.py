#!/usr/bin/env python

import sys
import os
import os.path


def main():
    TRAIN_TEXT_FILE = 'train.txt'
    VAL_TEXT_FILE = 'val.txt'
    IMAGE_FOLDER = '/home/karen/Downloads/data/dogvscats/train'

    fr = open(TRAIN_TEXT_FILE, 'w')
    fv = open(VAL_TEXT_FILE, 'w')

    filenames = os.listdir(IMAGE_FOLDER)
    for filename in filenames:
        if filename[0:3] == 'cat':
            if filename[-5] == '2':  # or filename[-5] == '8':
                fv.write(filename + ' 0\n')
            else:
                fr.write(filename + ' 0\n')
        if filename[0:3] == 'dog':
            if filename[-5] == '2':  # or filename[-5] == '8':
                fv.write(filename + ' 1\n')
            else:
                fr.write(filename + ' 1\n')

    fr.close()
    fv.close()


if __name__ == '__main__':
    main()
