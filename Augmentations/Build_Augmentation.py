from Augmentations.Augmentation import Augment


def build_augmentation(cfgs_augmentation):
    """
            Build an augmentation
            Args:
                cfgs_augmentation (eDICT):
            Returns:
                Augment: a constructed augmentation specified by augmentations.
    """
    return Augment(cfgs_augmentation)

