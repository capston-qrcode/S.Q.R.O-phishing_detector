# TODO : fix this to use sklearn
import tensorflow as tf

from keras.api import backend as K  # noqa
from keras.api.metrics import Metric


class F1Score(Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):  # TODO : fix this
        y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (
            self.true_positives + self.false_positives + K.epsilon()
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + K.epsilon()
        )
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
