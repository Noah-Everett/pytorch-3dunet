import os

import h5py
import numpy as np
import torch

from datasets.hdf5 import HDF5Dataset
from unet3d import utils
from unet3d.config import parse_test_config
from unet3d.model import UNet3D

logger = utils.get_logger('UNet3DPredictor')


def predict(model, dataset, out_channels, device):
    """
    Return prediction masks by applying the model on the given dataset

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        dataset (torch.utils.data.Dataset): input dataset
        out_channels (int): number of channels in the network output
        device (torch.Device): device to run the prediction on

    Returns:
         probability_maps (numpy array): prediction masks for given dataset
    """
    logger.info(f'Running prediction on {len(dataset)} patches...')
    # dimensionality of the the output (CxDxHxW)
    dataset_shape = dataset.raw.shape
    if len(dataset_shape) == 3:
        volume_shape = dataset_shape
    else:
        volume_shape = dataset_shape[1:]
    probability_maps_shape = (out_channels,) + volume_shape
    logger.info(f'The shape of the output probability maps (CDHW): {probability_maps_shape}')
    # initialize the output prediction array
    probability_maps = np.zeros(probability_maps_shape, dtype='float32')

    # initialize normalization mask in order to average out probabilities
    # of overlapping patches
    normalization_mask = np.zeros(probability_maps_shape, dtype='float32')

    # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
    model.eval()
    # Run predictions on the entire input dataset
    with torch.no_grad():
        for patch, index in dataset:
            logger.info(f'Predicting slice:{index}')

            # save patch index: (C,D,H,W)
            channel_slice = slice(0, out_channels)
            index = (channel_slice,) + index

            # convert patch to torch tensor NxCxDxHxW and send to device
            # we're using batch size of 1
            patch = patch.unsqueeze(dim=0).to(device)

            # forward pass
            probs = model(patch)
            # squeeze batch dimension and convert back to numpy array
            probs = probs.squeeze(dim=0).cpu().numpy()
            # unpad in order to avoid block artifacts in the output probability maps
            probs, index = utils.unpad(probs, index, volume_shape)
            # accumulate probabilities into the output prediction array
            probability_maps[index] += probs
            # count voxel visits for normalization
            normalization_mask[index] += 1

    return probability_maps / normalization_mask


def save_predictions(probability_maps, output_file, dataset_name='probability_maps'):
    """
    Saving probability maps to a given output H5 file. If 'average_channels'
    is set to True average the probability_maps across the the channel axis
    (useful in case where each channel predicts semantically the same thing).

    Args:
        probability_maps (numpy.ndarray): numpy array containing probability
            maps for each class in separate channels
        output_file (string): path to the output H5 file
        dataset_name (string): name of the dataset inside H5 file where the probability_maps will be saved
    """
    logger.info(f'Saving predictions to: {output_file}...')

    with h5py.File(output_file, "w") as output_h5:
        logger.info(f"Creating dataset '{dataset_name}'...")
        output_h5.create_dataset(dataset_name, data=probability_maps, compression="gzip")


def main():
    config = parse_test_config()

    # make sure those values correspond to the ones used during training
    in_channels = config.in_channels
    out_channels = config.out_channels
    # use F.interpolate for upsampling
    interpolate = config.interpolate
    layer_order = config.layer_order
    final_sigmoid = config.final_sigmoid
    model = UNet3D(in_channels, out_channels,
                   init_channel_number=config.init_channel_number,
                   final_sigmoid=final_sigmoid,
                   interpolate=interpolate,
                   conv_layer_order=layer_order)

    logger.info(f'Loading model from {config.model_path}...')
    utils.load_checkpoint(config.model_path, model)

    logger.info('Loading datasets...')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        logger.warning('No CUDA device available. Using CPU for prediction...')
        device = torch.device('cpu')

    model = model.to(device)

    patch = tuple(config.patch)
    stride = tuple(config.stride)

    for test_path in config.test_path:
        # create dataset for a given test file
        dataset = HDF5Dataset(test_path, patch, stride, phase='test', raw_internal_path=config.raw_internal_path)
        # run the model prediction on the entire dataset
        probability_maps = predict(model, dataset, out_channels, device)
        # save the resulting probability maps
        output_file = f'{os.path.splitext(test_path)[0]}_probabilities.h5'
        save_predictions(probability_maps, output_file)


if __name__ == '__main__':
    main()
