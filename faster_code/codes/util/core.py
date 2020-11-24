"""

"""

import tensorflow as tf
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
import matplotlib as plt


def non_linear_fn(t, x, y, z):
  # returns a function from variables
  return tf.nn.tanh(20*(t - 2*(x-.5)))+ tf.nn.selu((t-2*(y-0.5))) + tf.nn.sigmoid(-20*(t-(z-0.5)))


#"""[Summary]

#:param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
#:type [ParamName]: [ParamType](, optional)
#...
#:raises [ErrorType]: [ErrorDescription]
# ...
# :return: [ReturnDescription]
# :rtype: [ReturnType]
# """

def generate_data(values, function=non_linear_fn, length=25, range_=[-1, 1]):
    """
    Function to generate data from values

    :param values: values to function for generating spectra
    :type values: float
    :param function:  mathematical expression used to generate spectra
    :type function: function, optional
    :param length: spectral length
    :type length: int (optional)
    :param range_: x range for function
    :type range_:  list of float
    :return: generatered spectra
    :rtype: array of float
    """

    # build x vector
    x = np.linspace(range_[0], range_[1], 25)

    data = np.zeros((values.shape[0], 25))

    for i in range(values.shape[0]):
        data[i, :] = function(x, values[i, 0], values[i, 1], values[i, 2])

    return data

def find_nearest(array, value, averaging_number):
    """

    :param array: image to find the index closest to a value
    :type array: float, array
    :param value: value to find points near
    :type value: float
    :param averaging_number: number of points to find
    :type averaging_number: int
    :return: returns the indices nearest to a value in an image
    :rtype: array
    """

    idx = (np.abs(array-value)).argsort()[0:averaging_number]
    return idx

def rotate_and_crop(image_, angle=60.46, frac_rm=0.17765042979942694):
    """

    :param image_: image array to plot
    :type image_: array
    :param angle: angle to rotate the image by
    :type angle: float (, optional)
    :param frac_rm: sets the fraction of the image to remove
    :type frac_rm: float (, optional)
    :return: crop_image:
                 image which is rotated and cropped
             scale_factor:
                 scaling factor for the image following rotation
    :rtype: crop_image:
                 array
            scale_factor:
                 float
    """

    # makes a copy of the image
    image = np.copy(image_)
    # replaces all points with the minimum value
    image[~np.isfinite(image)] = np.nanmin(image)
    # rotates the image
    rot_topo = ndimage.interpolation.rotate(
        image, 90-angle, cval=np.nanmin(image))
    # crops the image
    pix_rem = int(rot_topo.shape[0]*frac_rm)
    crop_image = rot_topo[pix_rem:rot_topo.shape[0] -
                          pix_rem, pix_rem:rot_topo.shape[0]-pix_rem]
    # returns the scale factor for the new image size
    scale_factor = (np.cos(np.deg2rad(angle)) +
                    np.cos(np.deg2rad(90-angle)))*(1-frac_rm)

    return crop_image, scale_factor



def layout_fig(graph, mod=None):

    """

    :param graph: number of axes to make
    :type graph: int
    :param mod: sets the number of figures per row
    :type mod: int (, optional)
    :return: fig:
                handel to figure being created
             axes:
                numpy array of axes that are created
    :rtype: fig:
                matplotlib figure
            axes:
                numpy array
    """

    # Sets the layout of graphs in matplotlib in a pretty way based on the number of plots
    if mod is None:
        # Selects the number of columns to have in the graph
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        elif graph < 37:
            mod = 7

    # builds the figure based on the number of graphs and selected number of columns
    fig, axes = plt.subplots(graph // mod + (graph % mod > 0), mod,
                             figsize=(3 * mod, 3 * (graph // mod + (graph % mod > 0))))

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return (fig, axes)


def embedding_maps(data, image, colorbar_shown=True,
                   c_lim=None, mod=None,
                   title=None):

    """

    :param data:
    :type data: array
    :param image:
    :type image:
    :param colorbar_shown:
    :type colorbar_shown:
    :param c_lim:
    :type c_lim:
    :param mod:
    :type mod:
    :param title:
    :type title:
    :return: handel to figure being created
    :rtype: matplotlib figure
    """
    fig, ax = layout_fig(data.shape[1], mod)

    for i, ax in enumerate(ax):
        if i < data.shape[1]:
            im = ax.imshow(data[:, i].reshape(image.shape[0], image.shape[1]))
            ax.set_xticklabels('')
            ax.set_yticklabels('')

            # adds the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax, format='%.1e')

            # Sets the scales
            if c_lim is not None:
                im.set_clim(c_lim)

    if title is not None:
        # Adds title to the figure
        fig.suptitle(title, fontsize=16,
                     y=1, horizontalalignment='center')

    fig.tight_layout()

    return fig