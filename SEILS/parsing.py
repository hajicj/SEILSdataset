"""This module implements a class that..."""
from __future__ import print_function, unicode_literals

import logging

import collections
import os
import random

import numpy
import time

import pickle
from muscima.cropobject import cropobject_distance
from muscima.inference_engine_constants import _CONST
from muscima.io import parse_cropobject_list
from sklearn.feature_extraction import DictVectorizer

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################
# Classification wrapper


class PairwiseClassificationParser(object):
    """This parser applies a simple classifier that takes the bounding
    boxes of two CropObjects and their classes and returns whether there
    is an edge or not."""
    MAXIMUM_DISTANCE_THRESHOLD = 200

    def __init__(self, grammar, clf, cropobject_feature_extractor):
        self.grammar = grammar
        self.clf = clf
        self.extractor = cropobject_feature_extractor

    def parse(self, cropobjects):

        # Ensure the same docname for all cropobjects,
        # since we later compute their distances.
        # The correct docname gets set on export anyway.
        default_doc = cropobjects[0].doc
        for c in cropobjects:
            c.set_doc(default_doc)

        pairs, features = self.extract_all_pairs(cropobjects)

        logging.info('Clf.Parse: {0} object pairs from {1} objects'.format(len(pairs), len(cropobjects)))

        preds = self.clf.predict(features)

        edges = []
        for idx, (c_from, c_to) in enumerate(pairs):
            if preds[idx] != 0:
                edges.append((c_from.objid, c_to.objid))

        edges = self._apply_trivial_fixes(cropobjects, edges)
        return edges

    def _apply_trivial_fixes(self, cropobjects, edges):
        edges = self._only_one_stem_per_notehead(cropobjects, edges)
        edges = self._every_full_notehead_has_a_stem(cropobjects, edges)

        return edges

    def _only_one_stem_per_notehead(self, cropobjects, edges):
        _cdict = {c.objid: c for c in cropobjects}

        # Collect stems per notehead
        stems_per_notehead = collections.defaultdict(list)
        stem_objids = set()
        for f_objid, t_objid in edges:
            f = _cdict[f_objid]
            t = _cdict[t_objid]
            if (f.clsname in _CONST.NOTEHEAD_CLSNAMES) and \
                (t.clsname == 'stem'):
                stems_per_notehead[f_objid].append(t_objid)
                stem_objids.add(t_objid)

        # Pick the closest one (by minimum distance)
        closest_stems_per_notehead = dict()
        for n_objid in stems_per_notehead:
            n = _cdict[n_objid]
            stems = [_cdict[objid] for objid in stems_per_notehead[n_objid]]
            closest_stem = min(stems, key=lambda s: cropobject_distance(n, s))
            closest_stems_per_notehead[n_objid] = closest_stem.objid

        # Filter the edges
        edges = [(f_objid, t_objid) for f_objid, t_objid in edges
                 if (f_objid not in closest_stems_per_notehead) or
                    (t_objid not in stem_objids) or
                    (closest_stems_per_notehead[f_objid] == t_objid)]

        return edges

    def _every_full_notehead_has_a_stem(self, cropobjects, edges):
        _cdict = {c.objid: c for c in cropobjects}

        # Collect stems per notehead
        notehead_objids = set([c.objid for c in cropobjects if c.clsname == 'notehead-full'])
        stem_objids = set([c.objid for c in cropobjects if c.clsname == 'stem'])

        noteheads_with_stem_objids = set()
        stems_with_notehead_objids = set()
        for f, t in edges:
            if _cdict[f].clsname == 'notehead-full':
                if _cdict[t].clsname == 'stem':
                    noteheads_with_stem_objids.add(f)
                    stems_with_notehead_objids.add(t)

        noteheads_without_stems = {n: _cdict[n] for n in notehead_objids
                                      if n not in noteheads_with_stem_objids}
        stems_without_noteheads = {n: _cdict[n] for n in stem_objids
                                      if n not in stems_with_notehead_objids}

        # To each notehead, assign the closest stem that is not yet taken.
        closest_stem_per_notehead = {objid: min(stems_without_noteheads,
                                            key=lambda x: cropobject_distance(_cdict[x], n))
                                     for objid, n in noteheads_without_stems.items()}

        # Filter edges that are too long
        _n_before_filter = len(closest_stem_per_notehead)
        closest_stem_threshold_distance = 80
        closest_stem_per_notehead = {n_objid: s_objid
                                     for n_objid, s_objid in closest_stem_per_notehead.items()
                                     if cropobject_distance(_cdict[n_objid],
                                                            _cdict[s_objid])
                                         < closest_stem_threshold_distance
                                     }

        return edges + list(closest_stem_per_notehead.items())

    def extract_all_pairs(self, cropobjects):
        pairs = []
        features = []
        for u in cropobjects:
            for v in cropobjects:
                if u.objid == v.objid:
                    continue
                distance = cropobject_distance(u, v)
                if distance < self.MAXIMUM_DISTANCE_THRESHOLD:
                    pairs.append((u, v))
                    f = self.extractor(u, v)
                    features.append(f)

        # logging.info('Parsing features: {0}'.format(features[0]))
        features = numpy.array(features)
        # logging.info('Parsing features: {0}/{1}'.format(features.shape, features))
        return pairs, features

    def is_edge(self, c_from, c_to):
        features = self.extractor(c_from, c_to)
        result = self.clf.predict(features)
        return result

    def set_grammar(self, grammar):
        self.grammar = grammar


##############################################################################
# Feature extraction


class PairwiseClfFeatureExtractor:
    def __init__(self, vectorizer=None):
        """Initialize the feature extractor.

        :param vectorizer: A DictVectorizer() from scikit-learn.
            Used to convert feature dicts to the vectors that
            the edge classifier of the parser will expect.
            If None, will create a new DictVectorizer. (This is useful
            for training; you can then pickle the entire extractor
            and make sure the feature extraction works for the classifier
            at runtime.)
        """
        if vectorizer is None:
            vectorizer = DictVectorizer()
        self.vectorizer = vectorizer

    def __call__(self, *args, **kwargs):
        """The call is per item (in this case, CropObject pair)."""
        fd = self.get_features_relative_bbox_and_clsname(*args, **kwargs)
        # Compensate for the vecotrizer "target", which we don't have here (by :-1)
        item_features = self.vectorizer.transform(fd).toarray()[0, :-1]
        return item_features

    def get_features_relative_bbox_and_clsname(self, c_from, c_to):
        """Extract a feature vector from the given pair of CropObjects.
        Does *NOT* convert the class names to integers.

        Features: bbox(c_to) - bbox(c_from), clsname(c_from), clsname(c_to)
        Target: 1 if there is a link from u to v

        Returns a dict that works as input to ``self.vectorizer``.
        """
        target = 0
        if c_from.doc == c_to.doc:
            if c_to.objid in c_from.outlinks:
                target = 1
        features = (c_to.top - c_from.top,
                    c_to.left - c_from.left,
                    c_to.bottom - c_from.bottom,
                    c_to.right - c_from.right,
                    c_from.clsname,
                    c_to.clsname,
                    target)
        dt, dl, db, dr, cu, cv, tgt = features
        # Normalizing clsnames
        if cu.startswith('letter'): cu = 'letter'
        if cu.startswith('numeral'): cu = 'numeral'
        if cv.startswith('letter'): cv = 'letter'
        if cv.startswith('numeral'): cv = 'numeral'
        feature_dict = {'dt': dt,
                        'dl': dl,
                        'db': db,
                        'dr': dr,
                        'cls_from': cu,
                        'cls_to': cv,
                        'target': tgt}
        return feature_dict

    def get_features_distance_relative_bbox_and_clsname(self, c_from, c_to):
        """Extract a feature vector from the given pair of CropObjects.
        Does *NOT* convert the class names to integers.

        Features: bbox(c_to) - bbox(c_from), clsname(c_from), clsname(c_to)
        Target: 1 if there is a link from u to v

        Returns a tuple.
        """
        target = 0
        if c_from.doc == c_to.doc:
            if c_to.objid in c_from.outlinks:
                target = 1
        distance = cropobject_distance(c_from, c_to)
        features = (distance,
                    c_to.top - c_from.top,
                    c_to.left - c_from.left,
                    c_to.bottom - c_from.bottom,
                    c_to.right - c_from.right,
                    c_from.clsname,
                    c_to.clsname,
                    target)
        dist, dt, dl, db, dr, cu, cv, tgt = features
        if cu.startswith('letter'): cu = 'letter'
        if cu.startswith('numeral'): cu = 'numeral'
        if cv.startswith('letter'): cv = 'letter'
        if cv.startswith('numeral'): cv = 'numeral'
        feature_dict = {'dist': dist,
                        'dt': dt,
                        'dl': dl,
                        'db': db,
                        'dr': dr,
                        'cls_from': cu,
                        'cls_to': cv,
                        'target': tgt}
        return feature_dict


##############################################################################


# TODO: Move these params into some config/strategy object
# Data point sampling parameters
THRESHOLD_NEGATIVE_DISTANCE = 200
MAX_NEGATIVE_EXAMPLES_PER_OBJECT = None

# Feature extraction params

# Classifier hyperparameters
CLF_DTREE_MAX_DEPTH = 50
CLF_DTREE_MIN_SAMPLES_LEAF = 10


class PairwiseParsingStrategy(object):
    """Contains parameters of the training process: data point sampling,
    feature extraction, classifier."""

    def __init__(self,
                 max_object_distance=THRESHOLD_NEGATIVE_DISTANCE,
                 max_negative_samples_per_object=MAX_NEGATIVE_EXAMPLES_PER_OBJECT,
                 clf_dtree_max_depth=CLF_DTREE_MAX_DEPTH,
                 clf_dtree_min_samples_leaf=CLF_DTREE_MIN_SAMPLES_LEAF):
        """Only fills in the (hyper)parameter values.

        :param max_object_distance: Maximum distance over which objects can
            be connected.

        :param max_negative_samples_per_object:
        :param clf_dtree_max_depth:
        :param clf_dtree_min_samples_leaf:
        :return:
        """
        self.max_object_distance = max_object_distance
        self.max_negative_samples_per_object = max_negative_samples_per_object
        self.clf_dtree_max_depth = clf_dtree_max_depth
        self.clf_dtree_min_samples_leaf = clf_dtree_min_samples_leaf


##############################################################################


def symbol_distances(cropobjects):
    """For each pair of cropobjects, compute the closest distance between their
    bounding boxes.

    :returns: A dict of dicts, indexed by objid, then objid, then distance.
    """
    _start_time = time.clock()
    distances = {}
    for c in cropobjects:
        distances[c] = {}
        for d in cropobjects:

            if d not in distances:
                distances[d] = {}
            if d not in distances[c]:
                delta = cropobject_distance(c, d)
                distances[c][d] = delta
                distances[d][c] = delta
    print('Distances for {0} cropobjects took {1:.3f} seconds'
          ''.format(len(cropobjects), time.clock() - _start_time))
    return distances


def get_close_objects(dists, threshold=100):
    """Returns a dict: for each cropobject a list of cropobjects
    that are within the threshold."""
    output = {}
    for c in dists:
        output[c] = []
        for d in dists[c]:
            if dists[c][d] < threshold:
                output[c].append(d)
    return output


def negative_example_pairs(cropobjects,
                           threshold=THRESHOLD_NEGATIVE_DISTANCE,
                           max_per_object=MAX_NEGATIVE_EXAMPLES_PER_OBJECT):
    dists = symbol_distances(cropobjects)
    close_neighbors = get_close_objects(dists, threshold=threshold)
    # Exclude linked ones
    negative_example_pairs_dict = {}
    for c in close_neighbors:
        negative_example_pairs_dict[c] = [d for d in close_neighbors[c] if d.objid not in c.outlinks]

        # Downsample,
    # but intelligently: there should be more weight on closer objects, as they should
    # be represented more.
    if max_per_object is not None:
        for c in close_neighbors:
            random.shuffle(negative_example_pairs_dict[c])
            negative_example_pairs_dict[c] = negative_example_pairs_dict[c][:max_per_object]
    negative_example_pairs = []
    for c in negative_example_pairs_dict:
        negative_example_pairs.extend([(c, d) for d in negative_example_pairs_dict[c]])
    return negative_example_pairs


def positive_example_pairs(cropobjects):
    _cdict = {c.objid: c for c in cropobjects}
    positive_example_pairs = []
    for c in cropobjects:
        for o in c.outlinks:
            positive_example_pairs.append((c, _cdict[o]))
    return positive_example_pairs


def get_object_pairs(cropobjects,
                     max_object_distance=THRESHOLD_NEGATIVE_DISTANCE,
                     max_negative_samples=MAX_NEGATIVE_EXAMPLES_PER_OBJECT):

    negs = negative_example_pairs(cropobjects,
                                  threshold=max_object_distance,
                                  max_per_object=max_negative_samples)
    poss = positive_example_pairs(cropobjects)
    return negs + poss


def train_clf(mungs, strategy, do_eval=False):
    """Train the relationship classifier: a decision tree.

    :param mungs: The list of MuNG documents (list of lists of CropObjects).

    :param do_eval: Train the clf. on an 80-20 train/test split
        and perform evaluation (accuracy & positives f-score).

    :return: ``(vectorizer, classifier)``. Vectorizer maps feature
        names to columns (especially needed for class name features).
        The classifier is a sklearn DecisionTree.
    """
    feature_extractor = PairwiseClfFeatureExtractor()

    # Sample training object pairs & extract their features
    features = []
    for mung in mungs:
        object_pairs = get_object_pairs(
            mung,
            max_object_distance=strategy.max_object_distance,
            max_negative_samples=strategy.max_negative_samples_per_object)
        f = [feature_extractor.get_features_distance_relative_bbox_and_clsname(u, v)
             for u, v in object_pairs]
        features.extend(f)

    # Convert features through vectorizer (still Feature Extractor)
    feature_vectors = feature_extractor.vectorizer.fit_transform(features).toarray()

    # Prepare features as training/test data
    from sklearn.model_selection import train_test_split
    X = feature_vectors[:, :-1]
    y = feature_vectors[:, -1]
    if do_eval:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    else:
        X_train, y_train = X, y

    logging.info('Initializing classifier')
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=CLF_DTREE_MAX_DEPTH,
                                 min_samples_leaf=CLF_DTREE_MIN_SAMPLES_LEAF)

    logging.info('Fitting classifier with {} training data points'.format(X_train.shape[0]))
    clf.fit(X_train, y_train)

    if do_eval:
        logging.info('Evaluating classifier...')
        y_test_pred = clf.predict(X_test)
        from sklearn.metrics import f1_score, accuracy_score

        f1 = f1_score(y_test, y_test_pred)
        acc = accuracy_score(y_test, y_test_pred)
        print('F-score: {0:.3f}\nAccuracy: {1:.3f}'.format(f1, acc))

    return feature_extractor.vectorizer, clf


def create_parsing_model(mung_dir, output_dir, output_name, do_eval=False,
                         strategy=PairwiseParsingStrategy()):
    """Creates the vectorizer and parsing classifier and pickles
    them into the given directory.

    :param mung_dir: Input MuNG direcotory, from which the parser/vectorizer
        will be trained.

    :param output_dir: Output directory, into which the vectorizer
        and classifier will be pickled.

    :param output_tag: The root name of the vectorizer and classifier.
        The final names will be created by adding ``.vectorizer.pkl``
        and ``classifier.pkl``.

    :param strategy: Specify the (hyper)parameters of the parser through
        a ``PairwiseParsingStrategy`` object.
    """
    mungs = []
    for f in os.listdir(mung_dir):
        if not f.endswith('.xml'):
            continue
        mung = parse_cropobject_list(os.path.join(mung_dir, f))
        mungs.append(mung)

    vectorizer, clf = train_clf(mungs,
                                strategy=strategy,
                                do_eval=do_eval)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    vec_name = os.path.join(output_dir, output_name + '.vectorizer.pkl')
    with open(vec_name, 'wb') as hdl:
        pickle.dump(vectorizer, hdl, protocol=pickle.HIGHEST_PROTOCOL)

    clf_name = os.path.join(output_dir, output_name + '.classifier.pkl')
    with open(clf_name, 'wb') as hdl:
        pickle.dump(clf, hdl, protocol=pickle.HIGHEST_PROTOCOL)


def load_parser(parser_dir, parser_name):
    """Creates the PairwiseClassificationParser from the stored training
    results."""
    vec_name = os.path.join(parser_dir, parser_name + '.vectorizer.pkl')
    with open(vec_name, 'wb') as hdl:
        vectorizer = pickle.load(hdl)

    clf_name = os.path.join(parser_dir, parser_name + '.classifier.pkl')
    with open(clf_name, 'wb') as hdl:
        clf = pickle.load(hdl)

    feature_extractor = PairwiseClfFeatureExtractor(vectorizer=vectorizer)
    parser = PairwiseClassificationParser(grammar=None,
                                          clf=clf,
                                          cropobject_feature_extractor=feature_extractor)

    return parser


##############################################################################
