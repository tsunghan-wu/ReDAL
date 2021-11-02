from active_selection import random_selection, softmax_uncertainty, segment_entropy
from active_selection import core_set
from active_selection import mc_dropout
from active_selection import ReDAL


def get_active_selector(FLAGS, region=False):
    if region is False:
        if FLAGS.active_method == 'random':
            return random_selection.RandomSelector()
        elif FLAGS.active_method in ['softmax_confidence', 'softmax_margin', 'softmax_entropy']:
            return softmax_uncertainty.SoftmaxUncertaintySelector(8, 8, FLAGS.active_method)
        elif FLAGS.active_method == 'coreset':
            return core_set.CoreSetSelector(8, 8)
        elif FLAGS.active_method == 'segment_entropy':
            return segment_entropy.SegmentEntropySelector(8, 8)
        elif FLAGS.active_method == 'mc_dropout':
            return mc_dropout.MCDropoutSelector(8, 8)
    else:
        if FLAGS.active_method == 'random':
            return random_selection.RegionRandomSelector()
        elif FLAGS.active_method in ['softmax_confidence', 'softmax_margin', 'softmax_entropy']:
            return softmax_uncertainty.RegionSoftmaxUncertaintySelector(8, 8, FLAGS.active_method)
        elif FLAGS.active_method == 'mc_dropout':
            return mc_dropout.RegionMCDropoutSelector(8, 8)
        elif FLAGS.active_method == 'ReDAL':
            return ReDAL.ReDALSelector(8, 8, FLAGS.ReDAL_config_path)
