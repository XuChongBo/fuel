import os
import h5py
import numpy
from PIL import Image

from fuel.converters.base import check_exists, progress_bar
from fuel.datasets.hdf5 import H5PYDataset

from os.path import splitext, basename
import time
import traceback

def get_example_list(desc_file_path, relative=True):
    example_list = []
    import codecs
    with codecs.open(desc_file_path, 'r', 'utf-8') as infile:
        idx = 1
        for line in infile:
            components  = line.strip().split(' ')
            tag = components[0]
            if len(components)==2:
                path = components[1]
            else:
                path = "{}.png".format(idx)
                idx += 1
            if relative:
                path = os.path.join(os.path.dirname(desc_file_path), path)
            print path
            example_list.append((tag, path))
    return example_list


def convert_captcha(directory, output_directory,
                         output_filename='captcha.hdf5'):
    """Converts captcha dataset to HDF5.

    Converts captcha to an HDF5 dataset compatible with
    :class:`fuel.datasets.captcha`. 
    The converted dataset is saved as 'captcha.hdf5'.

    It assumes the existence of the directory:  
        ./captcha/lineImages
        ./captcha/ascii-all

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'captcha.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    # Prepare input 
    all_example_paths = get_example_list(os.path.join(directory,'captcha','ans.txt'))
    split = "all"
    split_size = len(all_example_paths)
    
    # Prepare output file
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')
    dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    hdf_features = h5file.create_dataset('image_features', (split_size,),
                                         dtype=dtype)
    hdf_shapes = h5file.create_dataset('image_features_shapes', (split_size, 3),
                                       dtype='int32')
    hdf_targets = h5file.create_dataset('targets', (split_size,), dtype=h5py.special_dtype(vlen=bytes))
    hdf_targets_shapes = h5file.create_dataset('hdf_targets_shapes', (split_size, 1),
                                       dtype='int32')
    # Attach shape annotations and scales
    hdf_features.dims.create_scale(hdf_shapes, 'shapes')
    hdf_features.dims[0].attach_scale(hdf_shapes)

    hdf_shapes_labels = h5file.create_dataset('image_features_shapes_labels',
                                              (3,), dtype='S7')
    hdf_shapes_labels[...] = ['channel'.encode('utf8'),
                              'height'.encode('utf8'),
                              'width'.encode('utf8')]
    hdf_features.dims.create_scale(hdf_shapes_labels, 'shape_labels')
    hdf_features.dims[0].attach_scale(hdf_shapes_labels)

    hdf_targets.dims.create_scale(hdf_targets_shapes, 'targets_shapes')
    hdf_targets.dims[0].attach_scale(hdf_targets_shapes)

    # Add axis annotations
    hdf_features.dims[0].label = 'batch'

    hdf_targets.dims[0].label = 'batch'
    #hdf_targets.dims[1].label = 'index'



    # Shuffle the examples
    rng = numpy.random.RandomState(123522)
    rng.shuffle(all_example_paths)

    # Convert from JPEG to NumPy arrays
    with progress_bar(split, split_size) as bar:
        i = 0
        for tag, image_path in all_example_paths:
            # Save image
            image = numpy.array(Image.open(image_path))
            print image.shape
            assert(image.ndim==3) #(height, width, channel)
            hdf_features[i] = image.flatten()
            hdf_shapes[i] = image.shape
            print image.shape

            #get the target
            textline = tag
            print textline
            hdf_targets[i] = textline #numpy.array(textline)
            hdf_targets_shapes[i] = len(textline)

            # Update progress
            i += 1
            bar.update(i)

    # Add the labels
    split_dict = {}
    sources = ['image_features', 'targets']
    split_dict['all'] = dict(zip(sources, [(0, split_size)] * 2))
    h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)


    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the captcha dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `captcha` command.

    """
    return convert_captcha
