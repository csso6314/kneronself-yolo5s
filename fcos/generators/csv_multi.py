# -*- coding: utf-8 -*-

from generators.generator import Generator
from utils.image import read_image_rgb

import numpy as np
from PIL import Image
from six import raise_from
import csv
import sys
import os.path
from collections import OrderedDict


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """
    Parse the classes file given by csv_reader.
    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_annotations(csv_reader, classes, base_dir):
    """
    Read annotations from the csv_reader.
    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader):
        line += 1
        if line == 1:
            continue
        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                None)
        img_file = base_dir + '/' + img_file
        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, float, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, float, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, float, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, float, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': classes[class_name]})
    return result


def _open_for_csv(path):
    """
    Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb', for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


class CSVGenerator(Generator):
    """
    Generate data for multiple custom json dataset.
    """

    def __init__(
            self,
            data_files,
            class_files,
            base_dirs,
            **kwargs
    ):
        """
        Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.image_names = []
        self.image_data = {}
        if isinstance(data_files, str):
            # list of str
            data_files = [data_files]
        if isinstance(class_files, str) or isinstance(class_files, dict) :
            # list of dict or str
            class_files = [class_files]
        if isinstance(base_dirs, str):
            # list of str
            base_dirs = [base_dirs]

        # parse the provided class file
        n_dataset = len(data_files)
        self.classes = []
        self.image_data = {}
        self.image_names = []

        for set_idx in range(n_dataset):
            # class_name --> class_id
            # [{'person': 0, ...}]
            class_file = class_files[set_idx]
            if isinstance(class_file, dict):
                # list of dict {str:int}
                self.classes.append(class_file)
            else:
                try:
                    with _open_for_csv(class_file) as file:
                        class_file = _read_classes(csv.reader(file, delimiter=','))
                        self.classes.append(class_file)
                except ValueError as e:
                    raise_from(ValueError('invalid CSV class file: {}: {}'.format(class_file, e)), None)

            data_file = data_files[set_idx]
            base_dir = base_dirs[set_idx]
            # csv with img_path, x1, y1, x2, y2, class_name
            try:
                with _open_for_csv(data_file) as file:
                    # {'img_path1':[{'x1':xx,'y1':xx,'x2':xx,'y2':xx,'class':xx}...],...}
                     image_data = _read_annotations(csv.reader(file, delimiter=','), class_file, base_dir)
                self.image_data.update(image_data)
            except ValueError as e:
                raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(data_file, e)), None)
        self.image_names = list(self.image_data.keys())

        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return np.max([item.values() for item in self.classes])+1

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        # self.labels 是 class_id --> class_name 的 dict
        return label < self.num_classes()

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        for i in range(len(self.classes)):
            if name in self.classes[i]:
                return True
        return False

    def name_to_label(self, name):
        print('do not support for multiple dataset')
        return name

    def label_to_name(self, label):
        """
        Map label to name.
        """
        print('do not support for multiple dataset')
        return label

    def image_path(self, image_index):
        """
        Returns the image path for image_index.
        """
        return self.image_names[image_index]

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        return read_image_rgb(self.image_path(image_index))

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        path = self.image_names[image_index]
        annotations = {'labels': np.empty((0,), dtype=np.int32), 'bboxes': np.empty((0, 4))}

        for idx, annot in enumerate(self.image_data[path]):
            annotations['labels'] = np.concatenate((annotations['labels'], [annot['class']]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(annot['x1']),
                float(annot['y1']),
                float(annot['x2']),
                float(annot['y2']),
            ]]))

        return annotations
