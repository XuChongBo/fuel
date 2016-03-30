import os
import h5py
import numpy
from PIL import Image

from fuel.converters.base import check_exists, progress_bar
from fuel.datasets.hdf5 import H5PYDataset

from os.path import splitext, basename
import time
import traceback

def get_path_list(root_dir):
    path_list = []
    i = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.tif'):
                i += 1
                image_path = os.path.join(root, file)
                print i, image_path
                path_list.append(image_path)
                # if i==1000:
                #     return path_list
    return path_list


def get_target(image_path):
    """
    ../lineImages/a02/a02-000/a02-000-06.tif

    h10/h10-415/h10-415z.txt
    lineImages/p09/p09-124/p09-124z-01.tif  
    p09-124z-02.tif  p09-124z-03.tif  p09-124z-04.tif  p09-124z-05.tif  p09-124z-06.tif

    ascii/p09/p09-124/p09-124z.txt
OCR:

CSR:

Then the two men came into the Office,
and Gus was introducing her. Dr. Horgan' .
Eye behind his spectacles were friendly
and his smile kind. "Dr. Pentland told me
about you, Miss Holme," he said,
shaking hands.

    """
    print image_path
    textline = "null"
    try:
        splits = image_path.split('/')
        subs = splits[-1].split("-")
        filename = subs[0]+'-'+subs[1]+".txt"
        index = int(subs[2][:-4])
        txt_path = os.path.join("./iam_ondb/ascii", splits[-3], splits[-2], filename)
        print txt_path, index
        f = open(txt_path)
        line_list = []
        is_begin = False
        for line in f.readlines():
            line = line.strip()  # kill of the \r\n at the end of line
            #print line, 
            if line[0:4]=="CSR:" and len(line)==4:
                is_begin = True
            elif is_begin and line!="":
                line_list.append(line)  
        textline = line_list[index-1]
    except :
        print traceback.format_exc()
        raw_input('press any key to continue.  assign null to textline.')

    #print len(line)
    #print line_list[index-1]
    #print len(line_list)
    return textline


def convert_iam_ondb(directory, output_directory,
                         output_filename='iam_ondb.hdf5'):
    """Converts iam_ondb dataset to HDF5.

    Converts iam_ondb to an HDF5 dataset compatible with
    :class:`fuel.datasets.iam_ondb`. 
    The converted dataset is saved as 'iam_ondb.hdf5'.

    It assumes the existence of the directory:  
        ./iam_ondb/lineImages
        ./iam_ondb/ascii-all

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'iam_ondb.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    # Prepare input 
    all_image_paths = get_path_list(directory)
    total_num = len(all_image_paths)
    test_num = int(total_num/4.0)
    train_num = total_num - test_num




    # Prepare output file
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')
    dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    hdf_features = h5file.create_dataset('image_features', (total_num,),
                                         dtype=dtype)
    hdf_shapes = h5file.create_dataset('image_features_shapes', (total_num, 3),
                                       dtype='int32')
    hdf_labels = h5file.create_dataset('targets', (total_num,), dtype=h5py.special_dtype(vlen=bytes))
    hdf_labels_shapes = h5file.create_dataset('hdf_labels_shapes', (total_num, 1),
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

    hdf_labels.dims.create_scale(hdf_labels_shapes, 'lables_shapes')
    hdf_labels.dims[0].attach_scale(hdf_labels_shapes)

    # Add axis annotations
    hdf_features.dims[0].label = 'batch'

    hdf_labels.dims[0].label = 'batch'
    #hdf_labels.dims[1].label = 'index'

    # Convert

    i = 0
    for split, split_size in zip(["train", "test"], [train_num, test_num]):
        # Shuffle the examples
        rng = numpy.random.RandomState(123522)
        if split=="train":
            image_paths = all_image_paths[0:train_num]
        else:
            image_paths = all_image_paths[train_num:]
        rng.shuffle(image_paths)

        # Convert from JPEG to NumPy arrays
        with progress_bar(split, split_size) as bar:
            for image_path in image_paths:
                # Save image
                image = numpy.array(Image.open(image_path))
                print image.shape
                if image.ndim!=2:
                    continue
                # Add a channels axis
                image = image[numpy.newaxis,:,:]
                
                hdf_features[i] = image.flatten()
                hdf_shapes[i] = image.shape
                print image.shape

                #get the target
                textline = get_target(image_path)
                print textline
                hdf_labels[i] = textline #numpy.array(textline)
                hdf_labels_shapes[i] = len(textline)

                # Update progress
                i += 1
                bar.update(i if split == "train" else i - train_num)

    # Add the labels
    split_dict = {}
    sources = ['image_features', 'targets']
    split_dict['train'] = dict(zip(sources, [(0, train_num)] * 2))
    split_dict['test'] = dict(zip(sources, [(train_num, total_num)] * 2))
    h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    # data = (('train', 'features', train_features),
    #         ('train', 'targets', train_labels),
    #         ('test', 'features', test_features),
    #         ('test', 'targets', test_labels))
    # fill_hdf5_file(h5file, data)
    # h5file['features'].dims[0].label = 'batch'
    # h5file['features'].dims[1].label = 'channel'
    # h5file['features'].dims[2].label = 'height'
    # h5file['features'].dims[3].label = 'width'

    # h5file['targets'].dims[0].label = 'batch'
    # h5file['targets'].dims[1].label = 'index'

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the iam_ondb dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `iam_ondb` command.

    """
    return convert_iam_ondb
