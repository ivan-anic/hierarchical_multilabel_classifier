# adapted from https://github.com/google-research/bert/blob/master/run_classifier.py


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self, input_ids, input_mask, segment_ids, label_ids, is_real_example=True
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = (label_ids,)
        self.is_real_example = is_real_example


def create_examples(df, labels_available=True):
    examples = []
    for (i, row) in enumerate(df.values):
        text_a = row[0]
        if labels_available:
            labels = row[1:]
        else:
            labels = np.zeros((len(row[1:]), 1))
        examples.append(InputExample(guid=i, text_a=text_a, labels=labels))
    return examples
