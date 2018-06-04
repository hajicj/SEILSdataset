#!/usr/bin/env python
"""This is a script that..."""
from __future__ import print_function, unicode_literals
import argparse
import itertools
import logging
import os
import time

import collections
import numpy
from scipy.misc import imread, imsave

from muscima.io import parse_cropobject_list

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--img_dir', action='store',
                        help='Input images directory. Copied over as "fulls".')
    parser.add_argument('--mung_dir', action='store',
                        help='Input MuNG directory. Assumes that the filenames'
                             ' correspond to the image filenames.')
    parser.add_argument('--output_masks', action='store',
                        help='Output root directory for exporting class masks.')
    parser.add_argument('--output_labels', action='store',
                        help='Output root directory for exporting class label'
                             ' images.')
    parser.add_argument('--export_fulls', action='store_true',
                        help='If set, will export the input images as the ``fulls``'
                             ' label.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    # Get list of MuNGs
    available_mung_names = [os.path.splitext(f)[0]
                            for f in os.listdir(args.mung_dir)
                            if f.endswith('.xml')]
    mungs = {f: parse_cropobject_list(os.path.join(args.mung_dir, f + '.xml'))
             for f in available_mung_names}

    # Get union of all labels. (In some images, a label might not exist, but
    #  we still want to export at least the black screen.)
    available_clsnames = set([c.clsname for c in itertools.chain.from_iterable(mungs.values())])

    # Create output directories
    if not os.path.isdir(args.output_masks):
        os.mkdir(args.output_masks)
    if not os.path.isdir(args.output_labels):
        os.mkdir(args.output_labels)
    for c in available_clsnames:
        c_mask_dir = os.path.join(args.output_masks, c)
        if not os.path.isdir(c_mask_dir):
            os.mkdir(c_mask_dir)
        c_labels_dir = os.path.join(args.output_labels, c)
        if not os.path.isdir(c_labels_dir):
            os.mkdir(c_labels_dir)

    if args.export_fulls:
        m_fulls_dir = os.path.join(args.output_masks, 'fulls')
        if not os.path.isdir(m_fulls_dir):
            os.mkdir(m_fulls_dir)
        l_fulls_dir = os.path.join(args.output_labels, 'fulls')
        if not os.path.isdir(l_fulls_dir):
            os.mkdir(l_fulls_dir)

    # Get list of images
    available_img_names = [os.path.splitext(f)[0]
                           for f in os.listdir(args.img_dir)
                           if (f.lower().endswith('jpg')
                               or (f.lower().endswith('png')))]

    # Get their intersection: both available MuNG and image.
    available_names = [f for f in available_mung_names
                       if f in available_img_names]

    # For each available MuNG/image pair:
    for f in available_names:
        print('Processing image: {}'.format(f))
        img_fpath = os.path.join(args.img_dir, f + '.png')
        if not os.path.isfile(img_fpath):
            img_fpath = img_fpath[:-3] + 'jpg'
        img = imread(img_fpath, mode='L')

        if args.export_fulls:
            m_full_file = os.path.join(m_fulls_dir, os.path.basename(img_fpath))
            imsave(m_full_file, img)
            l_full_file = os.path.join(l_fulls_dir, os.path.basename(img_fpath))
            imsave(l_full_file, img)

        img_h, img_w = img.shape
        mung = mungs[f]
        mung_dict = collections.defaultdict(list)
        for m in mung:
            mung_dict[m.clsname].append(m)

        # For each label:
        for c in available_clsnames:
            labels = numpy.zeros((img_h, img_w), dtype='uint16')
            c_mungs = mung_dict[c]
            for i, m in enumerate(c_mungs):
                label = i + 1
                labels[m.top:m.bottom, m.left:m.right] = label * m.mask

            # Export labels image
            output_labels_file = os.path.join(args.output_labels, c, f + '.png')
            imsave(output_labels_file, labels)

            # Export masks image
            output_mask_file = os.path.join(args.output_masks, c, f + '.png')
            mask = labels * 1
            mask[mask != 0] = 1
            imsave(output_mask_file, mask)

    _end_time = time.clock()
    logging.info('SEILS export_masks.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
