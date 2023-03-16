from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformer.modeling_mlm_v2 import RobertaForMaskFineTuning
from transformers import RobertaTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

from pathlib import Path

# LABEL_NUM = 30522
LABEL_NUM = 50265

csv.field_size_limit(sys.maxsize)

oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if oncloud:
    fh = logging.FileHandler('debug_layer_loss.log')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
logger = logging.getLogger()



def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class Sst2Processor_mlm(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, args):
        self.dataset_num = args.dataset_num
        self.data_seed = args.data_seed
        self.class_num = 2


        if self.data_seed == 100:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 13:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 21:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 42:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 87:
            self.label_words =  ["Ġpositive", "Ġnegative"]

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return self.label_words

    def get_class_num(self):
        """See base class."""
        return self.class_num


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue



            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

    def _create_examples_eval(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue

            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

class Sst5Processor_mlm(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, args):
        self.dataset_num = args.dataset_num
        self.data_seed = args.data_seed
        self.class_num = 5

        if self.data_seed == 100:
            self.label_words =  ["Ġextraordinary",\
                                 "Ġgreat",\
                                 "Ġenough",\
                                 "Ġboring",\
                                 "Ġawful"]
        elif self.data_seed == 87:
            self.label_words =  ["Ġextraordinary",\
                                 "Ġgreat",\
                                 "Ġenough",\
                                 "Ġboring",\
                                 "Ġawful"]
        elif self.data_seed == 42:
            self.label_words =  ["Ġextraordinary",\
                                 "Ġgreat",\
                                 "Ġenough",\
                                 "Ġboring",\
                                 "Ġawful"]
        elif self.data_seed == 21:
            self.label_words =  ["Ġextraordinary",\
                                 "Ġgreat",\
                                 "Ġenough",\
                                 "Ġboring",\
                                 "Ġawful"]
        elif self.data_seed == 13:
            self.label_words =  ["Ġextraordinary",\
                                 "Ġgreat",\
                                 "Ġenough",\
                                 "Ġboring",\
                                 "Ġawful"]




    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return self.label_words

    def get_class_num(self):
        """See base class."""
        return self.class_num


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue

            if line[-1].find('4') != -1:

                label = []
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label_i = line[-1]
                label_i = self.label_words[0]
                label.append(label_i)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif line[-1].find('3') != -1:

                label = []
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label_i = line[-1]
                label_i = self.label_words[1]
                label.append(label_i)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif line[-1].find('2') != -1:

                label = []
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label_i = line[-1]
                label_i = self.label_words[2]
                label.append(label_i)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif line[-1].find('1') != -1:

                label = []
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label_i = line[-1]
                label_i = self.label_words[3]
                label.append(label_i)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif line[-1].find('0') != -1:

                label = []
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label_i = line[-1]
                label_i = self.label_words[4]
                label.append(label_i)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))


        return examples

    def _create_examples_eval(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue


            if line[-1].find('4') != -1:

                label = []
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label_i = line[-1]
                label_i = self.label_words[0]
                label.append(label_i)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif line[-1].find('3') != -1:

                label = []
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label_i = line[-1]
                label_i = self.label_words[1]
                label.append(label_i)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif line[-1].find('2') != -1:

                label = []
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label_i = line[-1]
                label_i = self.label_words[2]
                label.append(label_i)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif line[-1].find('1') != -1:

                label = []
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label_i = line[-1]
                label_i = self.label_words[3]
                label.append(label_i)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif line[-1].find('0') != -1:

                label = []
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label_i = line[-1]
                label_i = self.label_words[4]
                label.append(label_i)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))




        return examples

class MrProcessor_mlm(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, args):
        self.dataset_num = args.dataset_num
        self.data_seed = args.data_seed
        self.class_num = 2



        if self.data_seed == 100:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 13:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 21:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 42:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 87:
            self.label_words =  ["Ġpositive", "Ġnegative"]


    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return self.label_words

    def get_class_num(self):
        """See base class."""
        return self.class_num


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue



            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

    def _create_examples_eval(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue

            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

class CrProcessor_mlm(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, args):
        self.dataset_num = args.dataset_num
        self.data_seed = args.data_seed
        self.class_num = 2



        if self.data_seed == 100:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 13:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 21:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 42:
            self.label_words =  ["Ġpositive", "Ġnegative"]
        elif self.data_seed == 87:
            self.label_words =  ["Ġpositive", "Ġnegative"]

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return self.label_words

    def get_class_num(self):
        """See base class."""
        return self.class_num


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue



            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

    def _create_examples_eval(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue

            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

class MpqaProcessor_mlm(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, args):
        self.dataset_num = args.dataset_num
        self.data_seed = args.data_seed
        self.class_num = 2

        if self.data_seed == 100:
            self.label_words =  ["Ġgood", "Ġbad"]
        elif self.data_seed == 13:
            self.label_words =  ["Ġgood", "Ġbad"]
        elif self.data_seed == 21:
            self.label_words =  ["Ġgood", "Ġbad"]
        elif self.data_seed == 42:
            self.label_words =  ["Ġgood", "Ġbad"]
        elif self.data_seed == 87:
            self.label_words =  ["Ġgood", "Ġbad"]

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return self.label_words

    def get_class_num(self):
        """See base class."""
        return self.class_num


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue

            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

    def _create_examples_eval(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue

            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

class SubjProcessor_mlm(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, args):
        self.dataset_num = args.dataset_num
        self.data_seed = args.data_seed
        self.class_num = 2




        if self.data_seed == 100:
            self.label_words =  ["Ġactual", "Ġindividual"]
        elif self.data_seed == 13:
            self.label_words =  ["Ġactual", "Ġindividual"]
        elif self.data_seed == 21:
            self.label_words =  ["Ġactual", "Ġindividual"]
        elif self.data_seed == 42:
            self.label_words =  ["Ġactual", "Ġindividual"]
        elif self.data_seed == 87:
            self.label_words =  ["Ġactual", "Ġindividual"]


    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return self.label_words

    def get_class_num(self):
        """See base class."""
        return self.class_num


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue



            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

    def _create_examples_eval(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue

            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

class TrecProcessor_mlm(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, args):
        self.dataset_num = args.dataset_num
        self.data_seed = args.data_seed
        self.class_num = 2

        if self.data_seed == 100:
            self.label_words =  ["Ġwonderful", "Ġbrilliant", "Ġfantastic", "Ġterrible", "Ġdone", "Ġdisappointing"]
        elif self.data_seed == 13:
            self.label_words =  ["Ġbrilliant", "Ġamazing", "Ġwonderful", "Ġnot", "Ġawful", "Ġterrible"]
        elif self.data_seed == 21:
            self.label_words =  ["Ġgreat", "Ġperfect", "Ġbrilliant", "Ġterrible", "Ġdisappointing", "Ġbad"]
        elif self.data_seed == 42:
            self.label_words =  ["Ġbeautiful", "Ġperfect", "Ġfantastic", "Ġterrible", "Ġawful", "Ġhilarious"]
        elif self.data_seed == 87:
            self.label_words =  ["Ġfantastic", "Ġexcellent", "Ġbeautiful", "Ġterrible", "Ġawful", "Ġworse"]

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return self.label_words

    def get_class_num(self):
        """See base class."""
        return self.class_num


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, 3 * i - 2)
            text_a = line[0]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[3]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

            label = []
            guid = "%s-%s" % (set_type, 3 * i - 1)
            text_a = line[0]
            label_i = self.label_words[1] if line[-1].find('1') != -1 else self.label_words[4]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

            label = []
            guid = "%s-%s" % (set_type, 3 * i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[2] if line[-1].find('1') != -1 else self.label_words[5]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

    def _create_examples_eval(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue

            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[3]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

class ColaProcessor_mlm(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, args):
        self.dataset_num = args.dataset_num
        self.data_seed = args.data_seed
        self.class_num = 2

        if self.data_seed == 100:
            self.label_words =  ["Ġgood", "Ġbad"]
        elif self.data_seed == 13:
            self.label_words =  ["Ġgood", "Ġbad"]
        elif self.data_seed == 21:
            self.label_words =  ["Ġgood", "Ġbad"]
        elif self.data_seed == 42:
            self.label_words =  ["Ġgood", "Ġbad"]
        elif self.data_seed == 87:
            self.label_words =  ["Ġgood", "Ġbad"]

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_eval(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return self.label_words

    def get_class_num(self):
        """See base class."""
        return self.class_num


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue

            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

    def _create_examples_eval(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            label = []
            if i == 0:
                continue

            label = []
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label_i = line[-1]
            label_i = self.label_words[0] if line[-1].find('1') != -1 else self.label_words[1]
            label.append(label_i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

def convert_examples_to_features_mlm_1(examples, max_seq_length,
                                       tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""



    features = []
    mask_index = []

    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) > max_seq_length - 5:
            i = []
            i.append(max_seq_length - 1)
            mask_index.append(i)
        else:
            i = []
            i.append(len(tokens_a) + 4)
            mask_index.append(i)


    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 5:
                tokens_a = tokens_a[:(max_seq_length - 5)]

        tokens = ["<s>"] + tokens_a + ["</s>"] + ["ĠIt"] + ["Ġis"] + ["<mask>"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            masked_label_ids = tokenizer.convert_tokens_to_ids(example.label)
            lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
            lm_label_array[mask_index[ex_index]] = masked_label_ids

        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=lm_label_array,
                          seq_length=seq_length))
    return features, mask_index

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "imdb":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids

def get_tensor_data_mlm(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids

def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

def do_eval_mlm_5(class_num, label_words, args, model, task_name, eval_dataloader,
                  device, output_mode, eval_labels, num_labels, masked_label_ids, mask_index, tokenizer):
    eval_loss = 0
    nb_eval_steps = 0
    val_acc = 0
    preds = []
    eval_num = 0

    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):

        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            logits, _ = model(input_ids, segment_ids, input_mask)

        if output_mode == "classification":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        preds_ = logits.detach().cpu().numpy()

        tmp = []
        for (p_index, p) in enumerate(preds_):
            p = np.array(p)
            mask_index_i = mask_index[p_index + eval_num]
            b = np.array(p[mask_index_i[0]])
            b = normalization(b)
            c = np.full(num_labels, dtype=np.float, fill_value=0)
            c[masked_label_ids] = b[masked_label_ids]
            tmp.append(c.tolist())



        preds_ = tmp

        tmp = []
        for (label_index, label) in enumerate(label_ids.detach().cpu().numpy()):
            mask_index_i = mask_index[label_index + eval_num]
            tmp.append(label[mask_index_i[0]])
        label_ = tmp

        if output_mode == "classification":
            preds_ = np.argmax(preds_, axis=1)
        elif output_mode == "regression":
            preds_ = np.squeeze(preds_)
        

        if class_num == 2:
            class_1_ids = tokenizer.convert_tokens_to_ids(label_words[int(len(label_words)/2) : ])
            class_2_ids = tokenizer.convert_tokens_to_ids(label_words[ : int(len(label_words)/2)])

            preds__ = []
            for i in preds_:
                preds__.append(1 if i in class_1_ids else 2)

            label__ = []
            for i in label_:
                label__.append(1 if i in class_1_ids else 2)

        elif class_num == 5:
            class_1_ids = tokenizer.convert_tokens_to_ids(label_words[ : int(len(label_words)/5)])
            class_2_ids = tokenizer.convert_tokens_to_ids(label_words[int(len(label_words)/5) : int(len(label_words)/5) * 2])
            class_3_ids = tokenizer.convert_tokens_to_ids(label_words[int(len(label_words)/5) * 2 : int(len(label_words)/5) * 3])
            class_4_ids = tokenizer.convert_tokens_to_ids(label_words[int(len(label_words)/5) * 3 : int(len(label_words)/5) * 4])
            class_5_ids = tokenizer.convert_tokens_to_ids(label_words[int(len(label_words)/5) * 4 : ])
        
            preds__ = []
            for i in preds_:
                if i in class_1_ids:
                    preds__.append(1)
                elif i in class_2_ids:
                    preds__.append(2)
                elif i in class_3_ids:
                    preds__.append(3)
                elif i in class_4_ids:
                    preds__.append(4)
                elif i in class_5_ids:
                    preds__.append(5)

            label__ = []
            for i in label_:
                if i in class_1_ids:
                    label__.append(1)
                elif i in class_2_ids:
                    label__.append(2)
                elif i in class_3_ids:
                    label__.append(3)
                elif i in class_4_ids:
                    label__.append(4)
                elif i in class_5_ids:
                    label__.append(5)


        assert len(preds__) == len(label__)
        batch_acc = simple_accuracy(np.array(preds__), np.array(label__))

        val_acc += batch_acc * len(preds_)

        eval_num += len(preds_)

    val_acc /= eval_num

    eval_loss = eval_loss / nb_eval_steps

    result = {"acc": val_acc}
    result['eval_loss'] = eval_loss

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_mlm",
                        action='store_true',
                        help="Whether to run eval_mlm on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--aug_train',
                        action='store_true')
    parser.add_argument('--eval_step',
                        type=int,
                        default=50)
    parser.add_argument('--pred_distill',
                        action='store_true')
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)

    parser.add_argument('--pred_distill_multi_loss',
                        action='store_true')

    parser.add_argument("--dataset_num", default=16, type=int)
    parser.add_argument('--use_CLS', action='store_true')

    parser.add_argument("--data_seed", type=int)

    parser.add_argument("--model_frozen", action='store_true')


    parser.add_argument("--data_url", type=str, default="", help="s3 url")
    parser.add_argument("--train_url", type=str, default="", help="s3 url")
    parser.add_argument("--init_method", default='', type=str)


    args = parser.parse_args()


    if oncloud:
        os.environ['DLS_LOCAL_CACHE_PATH'] = "/cache"
        local_data_dir = os.environ['DLS_LOCAL_CACHE_PATH']
        assert mox.file.exists(local_data_dir)
        logging.info("local disk: " + local_data_dir)
        logging.info("copy data from s3 to local")
        logging.info(mox.file.list_directory(args.data_url, recursive=True))
        mox.file.copy_parallel(args.data_url, local_data_dir)
        logging.info("copy finish...........")


        args.student_model = os.path.join(local_data_dir, args.student_model)
        args.data_dir = os.path.join(local_data_dir, args.data_dir)
        if args.ood_eval:
            args.ood_data_dir = os.path.join(local_data_dir, args.ood_data_dir)

        args.output_dir = os.path.join(local_data_dir, args.output_dir)



    logger.info('The args: {}'.format(args))

    processors = {
        "sst-2": Sst2Processor_mlm,
        "sst-5": Sst5Processor_mlm,
        "mr": MrProcessor_mlm,
        "cr": CrProcessor_mlm,
        "mpqa": MpqaProcessor_mlm,
        "subj": SubjProcessor_mlm,
        "trec": TrecProcessor_mlm,
        "cola": ColaProcessor_mlm
    }

    output_modes = {
        "sst-2": "classification",
        "sst-5": "classification",
        "mr": "classification",
        "cr": "classification",
        "mpqa": "classification",
        "subj": "classification",
        "trec": "classification",
        "cola": "classification"
    }

    default_params = {
        "sst-2": {"num_train_epochs": 10, "max_seq_length": 256},
        "sst-5": {"num_train_epochs": 10, "max_seq_length": 256},
        "mr": {"num_train_epochs": 10, "max_seq_length": 256},
        "cr": {"num_train_epochs": 10, "max_seq_length": 256},
        "mpqa": {"num_train_epochs": 10, "max_seq_length": 256},
        "subj": {"num_train_epochs": 10, "max_seq_length": 256},
        "trec": {"num_train_epochs": 10, "max_seq_length": 256},
        "cola": {"num_train_epochs": 10, "max_seq_length": 256}
    }

    acc_tasks = ["sst-2", "sst-5", "mr", "cr", "mpqa", "subj", "trec", "cola"]
    corr_tasks = ["sts-b"]
    mcc_tasks = []

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name in default_params:
        args.max_seq_length = default_params[task_name]["max_seq_length"]


    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name](args)
    output_mode = output_modes[task_name]

    tokenizer = RobertaTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    few_shot_train_examples = processor.get_train_examples(args.data_dir)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    num_train_optimization_steps = int(
        len(few_shot_train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    few_shot_train_features, few_shot_train_mask_index = convert_examples_to_features_mlm_1(few_shot_train_examples, args.max_seq_length, tokenizer, output_mode)
    few_shot_train_data, _ = get_tensor_data_mlm(output_mode, few_shot_train_features)
    few_shot_train_sampler = RandomSampler(few_shot_train_data)
    few_shot_train_dataloader = DataLoader(few_shot_train_data, sampler=few_shot_train_sampler, batch_size=args.train_batch_size)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features, eval_mask_index = convert_examples_to_features_mlm_1(eval_examples, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data_mlm(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    test_examples = processor.get_test_examples(args.data_dir)
    test_features, test_mask_index = convert_examples_to_features_mlm_1(test_examples, args.max_seq_length, tokenizer, output_mode)
    test_data, test_labels = get_tensor_data_mlm(output_mode, test_features)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    masked_label_ids = tokenizer.convert_tokens_to_ids(processor.get_labels())
    label_words = processor.get_labels()
    class_num = processor.get_class_num()

    student_model = RobertaForMaskFineTuning.from_pretrained(args.student_model)
    student_model.to(device)

    if args.model_frozen:
        for i in range(student_model.config.num_hidden_layers):
            for name, param in student_model.encoder.layer[i].named_parameters():
                param.requires_grad = False

    if args.do_eval_mlm:
        logger.info("***** Running evaluation MLM *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        result = do_eval_mlm_5(class_num, label_words, args, student_model, task_name, eval_dataloader,
                               device, output_mode, eval_labels, LABEL_NUM, masked_label_ids, eval_mask_index, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    else:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(few_shot_train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        schedule = 'warmup_linear'
        if not args.pred_distill:
            schedule = 'none'
        optimizer = BertAdam(optimizer_grouped_parameters,
                            schedule=schedule,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)
        loss_mse = MSELoss()

        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        global_step = 0
        best_dev_acc = 0.0
        best_ood_dev_acc = 0.0
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        output_test_file = os.path.join(args.output_dir, "test_results.txt")

        epoch_num = 0
        step_num = 0

        flag_feature_optimizer = True
        flag_prediction_optimizer = True

        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch_num += 1
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(few_shot_train_dataloader, desc="Iteration", ascii=True)):

                student_model.train()

                step_num += 1
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                cls_loss = 0.

                cls_loss = student_model(input_ids, segment_ids, input_mask, masked_lm_labels = label_ids)
                loss = cls_loss

                tr_cls_loss += cls_loss.item()

                logger.info("***** LOSS printing *****")
                logger.info("loss")
                logger.info(loss)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (global_step + 1) % args.eval_step == 0:
                    logger.info("***** Running evaluation MLM *****")
                    logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    student_model.eval()

                    loss = tr_loss / (step + 1)
                    cls_loss = tr_cls_loss / (step + 1)

                    result = {}


                    result = do_eval_mlm_5(class_num, label_words, args, student_model, task_name, eval_dataloader,
                                           device, output_mode, eval_labels, LABEL_NUM, masked_label_ids, eval_mask_index, tokenizer)
                    result['global_step'] = global_step
                    result['cls_loss'] = cls_loss
                    result['loss'] = loss

                    result_to_file(result, output_eval_file)

                    save_model = False

                    if task_name in acc_tasks and result['acc'] > best_dev_acc:
                        best_dev_acc = result['acc']
                        save_model = True

                    if task_name in corr_tasks and result['corr'] > best_dev_acc:
                        best_dev_acc = result['corr']
                        save_model = True

                    if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                        best_dev_acc = result['mcc']
                        save_model = True

                    if save_model:

                        logger.info("***** Save model *****")
                        logger.info("***** Test Dataset Eval Result *****")

                        result = do_eval_mlm_5(class_num, label_words, args, student_model, task_name, test_dataloader,
                                            device, output_mode, test_labels, LABEL_NUM, masked_label_ids, test_mask_index, tokenizer)
                        result['global_step'] = global_step
                        result['cls_loss'] = cls_loss
                        result['loss'] = loss

                        result_to_file(result, output_test_file)

                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                        '''
                        model_name = "epoch_num_{}_global_step_{}_{}".format(epoch_num, global_step, WEIGHTS_NAME)
                        output_model_file = os.path.join(args.output_dir, model_name)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)
                        '''
                        

                        if oncloud:
                            logging.info(mox.file.list_directory(args.output_dir, recursive=True))
                            mox.file.copy_parallel(args.output_dir, args.train_url)
                        

if __name__ == "__main__":
    main()

