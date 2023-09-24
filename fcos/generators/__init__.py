from augmentor import train_aug, val_aug


def create_generators(input_size=512,
                      batch_size=16,
                      data_root=None,
                      data_root_val=None,
                      dataset_type='pascal',
                      data_augmentation_chain=None,
                      annotations_path=None,
                      val_annotations_path=None,
                      classes_path=None,
                      center_sampling_radius=1.,
                      anchor_param=None):
    """
    Create generators for training and validation.

    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': batch_size,
        'input_size': input_size,
        'center_sampling_radius': center_sampling_radius,
        'anchor_param': anchor_param
    }

    if data_augmentation_chain is None:
        data_augmentation_chain = train_aug(input_size=input_size)
    data_augmentation_chain2 = val_aug(input_size=input_size)
    if data_root_val is None:
        data_root_val = data_root

    if dataset_type == 'pascal':
        if annotations_path is None:
            annotations_path = 'trainval'
        if val_annotations_path is None:
            val_annotations_path = 'test'

        from generators.pascal import PascalVocGenerator
        train_generator = PascalVocGenerator(
            data_root,
            annotations_path,
            skip_difficult=True,
            transformations=data_augmentation_chain,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            data_root_val,
            val_annotations_path,
            skip_difficult=True,
            shuffle_groups=False,
            transformations=data_augmentation_chain2,
            **common_args
        )

    elif dataset_type == 'widerperson':
        if annotations_path is None:
            annotations_path = 'train'
        if val_annotations_path is None:
            val_annotations_path = 'val'

        from generators.widerperson import PersonGenerator
        train_generator = PersonGenerator(
            data_root,
            annotations_path,
            transformations=data_augmentation_chain,
            **common_args
        )

        validation_generator = PersonGenerator(
            data_root_val,
            val_annotations_path,
            shuffle_groups=False,
            transformations=data_augmentation_chain2,
            **common_args
        )
    elif dataset_type == 'csv':
        # class_name, class_id
        from generators.csv_ import CSVGenerator
        # classes_path = {'1':0}
        assert annotations_path is not None
        if classes_path is None:
            import warnings
            warnings.warn(
                'no class file provided, use {int:int}')
            classes_path = dict(zip(list(range(200)), list(range(200))))
        train_generator = CSVGenerator(
            annotations_path,
            classes_path,
            base_dir=data_root,
            transformations=data_augmentation_chain,
            **common_args
        )

        if val_annotations_path:
            validation_generator = CSVGenerator(
                val_annotations_path,
                classes_path,
                base_dir=data_root_val,
                shuffle_groups=False,
                transformations=data_augmentation_chain2,
                **common_args
            )
        else:
            validation_generator = None
    elif dataset_type == 'csvs':
        # class_name, class_id
        from generators.csv_multi import CSVGenerator
        # classes_path = {'1':0}
        assert annotations_path is not None
        assert classes_path is not None
        train_generator = CSVGenerator(
            annotations_path,
            classes_path,
            base_dirs=data_root,
            transformations=data_augmentation_chain,
            **common_args
        )

        if val_annotations_path:
            validation_generator = CSVGenerator(
                val_annotations_path,
                classes_path,
                base_dirs=data_root_val,
                shuffle_groups=False,
                transformations=data_augmentation_chain2,
                **common_args
            )
        else:
            validation_generator = None
    elif dataset_type == 'coco':
        from generators.coco import CocoGenerator
        if annotations_path is None:
            annotations_path = 'train2017'

        if val_annotations_path is None:
            val_annotations_path = 'val2017'
        train_generator = CocoGenerator(
            data_root,
            annotations_path,
            transformations=data_augmentation_chain,
            **common_args
        )

        validation_generator = CocoGenerator(
            data_root_val,
            val_annotations_path,
            shuffle_groups=False,
            transformations=data_augmentation_chain2,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(dataset_type))

    return train_generator, validation_generator
