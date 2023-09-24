"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

import cv2
from generators.generator import Generator
import numpy as np
import os
from six import raise_from
import xml.etree.ElementTree as ET
from utils.image import read_image_rgb

widerperson_classes = {
    '1': 0,
    '2': 0,
    '3': 1,
#     '4': 0,
    '5': 1
}


class PersonGenerator(Generator):
    """
    Generate data for a Pascal VOC dataset.

    See http://host.robots.ox.ac.uk/pascal/VOC/ for more information.
    """

    def __init__(
            self,
            data_dir,
            set_name,
            classes=widerperson_classes,
            image_extension='.jpg',
            **kwargs
    ):
        """
        Initialize a Pascal VOC data generator.

        Args:
            data_dir: the path of directory which contains ImageSets directory
            set_name: test|trainval|train|val
            classes: class names tos id mapping
            image_extension: image filename ext
            **kwargs:
        """
        self.data_dir = data_dir
        self.set_name = set_name
        self.classes = classes
        self.image_names = [l.strip().split(None, 1)[0] for l in
                            open(os.path.join(data_dir, set_name + '.txt')).readlines()]
        self.image_extension = image_extension
        # class ids to names mapping
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(PersonGenerator, self).__init__(**kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """
        Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        path = os.path.join(self.data_dir, 'Images', self.image_names[image_index] + self.image_extension)
        image = cv2.imread(path)
        h, w = image.shape[:2]
        return float(w) / float(h)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        path = os.path.join(self.data_dir, 'Images', self.image_names[image_index] + self.image_extension)
        image = read_image_rgb(path)
        return image

    def __parse_annotation(self, line):
        """
        Parse an annotation given an XML element.
        """
        element = line.strip().split()
        class_name = element[0]
        if class_name not in self.classes:
            return None, None
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((4,))
        label = self.name_to_label(class_name)

        box[0] = element[1]
        box[1] = element[2]
        box[2] = element[3]
        box[3] = element[4]
        return box, label

    def __parse_annotations(self, lines):
        """
        Parse all annotations under the xml_root.
        """
        annotations = {'labels': np.empty((0,), dtype=np.int32),
                       'bboxes': np.empty((0, 4))}
        for i, line in enumerate(lines):
            try:
                box, label = self.__parse_annotation(line)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            if box is not None:
                annotations['bboxes'] = np.concatenate([annotations['bboxes'], [box]])
                annotations['labels'] = np.concatenate([annotations['labels'], [label]])
        return annotations

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        filename = self.image_names[image_index] + self.image_extension + '.txt'
        try:
            lines = open(os.path.join(self.data_dir, 'Annotations', filename)).readlines()
            if len(lines)<=1:
                lines = []
            else:
                lines = lines[1:]
            return self.__parse_annotations(lines)
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)


if __name__ == '__main__':
    from augmentor.misc import MiscEffect
    from augmentor.color import VisualEffect

    misc_effect = MiscEffect(border_value=0)
    visual_effect = VisualEffect()

    generator = PersonGenerator(
        'datasets/VOC0712',
        'trainval',
        skip_difficult=True,
        misc_effect=misc_effect,
        visual_effect=visual_effect,
        batch_size=1
    )
    for inputs, targets in generator:
        print('hi')
