#!/usr/bin/env python
"""This is a script that takes SEILS mensural primitives annotation
and adds composite note symbols: longa, brevis, semibrevis, minima,
semiminima, open/closed."""
from __future__ import print_function, unicode_literals
import argparse
import logging
import time

from muscima.io import parse_cropobject_list, export_cropobject_list
from muscima.graph import NotationGraph
from muscima.inference_engine_constants import _CONST
from muscima.cropobject import cropobjects_merge_multiple

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################
# Constants


SEILS_COMPOSITES_MAPPING = {
    ('notehead-square', 'stem'): 'longa',
    ('notehead-square',): 'brevis',
    ('notehead-empty',): 'semibrevis',
    ('notehead-empty', 'stem'): 'minima',
    ('8th_flag', 'notehead-empty', 'stem'): 'semiminima',
    ('16th_flag', 'notehead-empty', 'stem'): 'semiminima',
    ('notehead-square-full',): 'coloured_brevis',
    ('notehead-full',): 'coloured_semibrevis',
    ('notehead-full', 'stem'): 'coloured_minima',
    ('8th_flag', 'notehead-full', 'stem'): 'coloured_semiminima',
    ('16th_flag', 'notehead-full', 'stem'): 'coloured_semiminima',
}


SEILS_RELEVANT_PRIMITIVES = set(reduce(lambda x, y: list(x) + list(y),
                                       SEILS_COMPOSITES_MAPPING.keys(),
                                       []))


##############################################################################


def build_composite(notehead, graph, objid):
    """Create a composite from the given notehead.

    :param notehead: Root notehead of the composite object.

    :param graph: The notation graph of the whole document.

    :param connect_notehead: If set, will add an edge from
        the composite to the notehead. This is meant for
        the situation where the primitives are retained.

    :return: The composite CropObject.
    """
    children = graph.children(notehead, classes=SEILS_RELEVANT_PRIMITIVES)
    composite_key = tuple(sorted([c.clsname for c in children + [notehead]]))
    composite_class = SEILS_COMPOSITES_MAPPING[composite_key]

    composite = cropobjects_merge_multiple([notehead] + children,
                                           clsname=composite_class,
                                           objid=objid)


    return composite


def build_SEILS_composites(cropobjects, retain_primitives=False,
                           connect_composite_to_notehead=False):
    """Transforms the list of MuNG objects obtained from SEILS primitives
    annotation into the composite mensural note objects.

    :param retain_primitives: If set, will not remove the primitives that
        build up the composite objects.

    :param connect_composite_to_notehead: If set, the composite object
        will have an edge leading to the notehead. This is done only
        if ``retain_primitives`` is set as well.
    """
    g = NotationGraph(cropobjects)

    composites = []

    # Each notehead participates in at least one composite.
    noteheads = [c for c in cropobjects if c.clsname in _CONST.NOTEHEAD_CLSNAMES]
    next_objid = max([c.objid for c in cropobjects]) + 1
    for n in noteheads:

        composite = build_composite(n,
                                    graph=g,
                                    objid=next_objid)

        # Graph updates
        # =============
        #
        # If the primitives are not retained, re-hang edges to/from
        # primitives on the composite object.
        if not retain_primitives:
            children = g.children(n, classes=SEILS_RELEVANT_PRIMITIVES)
            _internal_objids = set([c.objid for c in children + [n]])
            for o_objid in composite.outlinks:
                c = g._cdict[o_objid]
                c.inlinks = [i for i in c.inlinks
                             if i not in _internal_objids] + [composite.objid]
            for i_objid in composite.inlinks:
                c = g._cdict[i_objid]
                c.outlinks = [o for o in c.outlinks
                              if o not in _internal_objids] +  [composite.objid]
        # Else just update all of the composite's child/parent nodes.
        else:
            for o_objid in composite.outlinks:
                g._cdict[o_objid].inlinks.append(composite.objid)
            for i_objid in composite.inlinks:
                g._cdict[i_objid].outlinks.append(composite.objid)

        if connect_composite_to_notehead:
            # Connect notehead and composite.
            output_composite.outlinks.append(n.objid)
            n.inlinks.append(output_composite.objid)

        composites.append(composite)
        next_objid += 1

    # Build correct output
    if retain_primitives:
        output = cropobjects + composites
    else:
        inert_cropobjects = [c for c in cropobjects
                             if c.clsname not in SEILS_RELEVANT_PRIMITIVES]
        output = inert_cropobjects + composites

    return output


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input_mung', action='store',
                        help='Input MuNG file.')
    parser.add_argument('-o', '--output_mung', action='store',
                        help='Output MuNG file.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    cropobjects = parse_cropobject_list(args.input_mung)

    output_cropobjects = build_SEILS_composites(cropobjects)

    xml = export_cropobject_list(output_cropobjects)
    with open(args.output_mung, 'w') as hdl:
        hdl.write(xml)
        hdl.write('\n')

    _end_time = time.clock()
    logging.info('build_composites.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
