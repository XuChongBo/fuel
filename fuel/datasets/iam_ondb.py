from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path


class IAM_ONDB(H5PYDataset):
    """The iam_ondb of online images.

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' and 'test'.
    Notes
    -----
    Users can create their own
    training / validation split using the `subset` argument.

    """
    filename = 'iam_ondb.hdf5'

    default_transformers = uint8_pixels_to_floatX(('image_features',))
    
    def __init__(self, which_sets, **kwargs):
        super(IAM_ONDB, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)
