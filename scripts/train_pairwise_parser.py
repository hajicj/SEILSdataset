#!/usr/bin/env python
"""This is a script that trains a pairwise CropObject parser
from a MuNG dataset."""
from __future__ import print_function, unicode_literals
import argparse
import logging
import time

from SEILS.parsing import create_parsing_model, PairwiseParsingStrategy

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--mung_dir', action='store',
                        help='Directory that contains the MuNG files from'
                             ' which to train the parser.')
    parser.add_argument('--output_dir', action='store',
                        help='Directory to which to store the output parsing'
                             ' classifier and vectorizer.')
    parser.add_argument('--name', action='store',
                        help='How you want to name this parser.')

    parser.add_argument('--do_eval', action='store_true',
                        help='If set, will also evaluate the pairwise'
                             ' classifier on a heldout 0.2 of training'
                             ' data.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    strategy = PairwiseParsingStrategy()

    # Your code goes here
    create_parsing_model(mung_dir=args.mung_dir,
                         output_dir=args.output_dir,
                         output_name=args.name,
                         do_eval=args.do_eval,
                         strategy=strategy)

    _end_time = time.clock()
    logging.info('train_pairwise_parser.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
