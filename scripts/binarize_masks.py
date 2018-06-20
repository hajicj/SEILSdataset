#!/usr/bin/env python
"""This is a script that processes a MuNG file and its corresponding image,
takes all objects that have all-1 masks (assumes they were created by some
automatic means), and tries to guess the precise mask by binarizing
the corresponding image crop. Outputs a MuNG file."""
from __future__ import print_function, unicode_literals
import argparse
import logging
import time

import cv2

from scipy.misc import imread
from muscima.io import parse_cropobject_list, export_cropobject_list

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


def binarize_mask(m, img, inplace=True, margin=0, debug=True):
    """
    :param m:
    :param img:
    :param inplace:
    :param margin: Take this many pixels around the cropobject
        into account when computing binarization threshold.

    :return: The modified cropobject.
    """
    t, l, b, r = m.bounding_box
    crop = img[t:b, l:r]
    thr, crop_binary = cv2.threshold(crop, 0, 255, cv2.THRESH_OTSU)
    crop_binary = (crop_binary == 0).astype('uint8')

    if debug:
        import matplotlib.pyplot as plt
        plt.imshow(crop_binary, cmap='gray', interpolation='nearest')
        plt.title('Obj {}: threshold {}'.format(m.objid, thr))
        plt.show()
    if inplace:
        m.mask = crop_binary
    else:
        raise NotImplementedError('Mask binarization currently only works'
                                  ' in-place. Some clone_cropobject method'
                                  ' must be implemented first.')
    return m


##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-m', '--mung', action='store', required=True,
                        help='Input MuNG file.')
    parser.add_argument('-i', '--image', action='store', required=True,
                        help='Input image file.')
    parser.add_argument('-o', '--output_mung', action='store', required=True,
                        help='Output result to this file.')
    parser.add_argument('-c', '--classes', nargs='+', default=None, action='store',
                        help='(Optional) Specify which classes of MuNGOs'
                             ' should be processed.')
    parser.add_argument('--force_binarize', action='store_true',
                        help='If set, will force binarization over all MuNGOs,'
                             'even those that have non-trivial masks.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    mungs = parse_cropobject_list(args.mung)

    img = imread(args.image, mode='L')

    output_mungs = []
    for m in mungs:
        if (args.classes is not None) and (m.clsname not in args.classes):
            output_mungs.append(m)
            continue
        if (not args.force_binarize) and (m.mask.nonzero()[0].shape[0] == 0):
            output_mungs.append(m)
            continue
        output_mungs.append(binarize_mask(m, img, inplace=True))

    xml = export_cropobject_list(output_mungs)
    with open(args.output_mung, 'w') as hdl:
        hdl.write(xml)
        hdl.write('\n')

    _end_time = time.clock()
    logging.info('binarize_masks.py done in {0:.3f} s'.format(_end_time - _start_time))


##############################################################################


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
