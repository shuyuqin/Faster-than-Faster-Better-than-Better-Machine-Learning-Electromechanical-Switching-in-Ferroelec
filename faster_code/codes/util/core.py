"""

"""

import tensorflow as tf
import numpy as np
from moviepy.tools import verbose_print
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning
from tqdm import tqdm
from .file import *
from .machine_learning import *
from natsort import natsorted, ns

printing = {'PNG':True,
            'EPS':False,
           'dpi': 300}

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
    x = np.linspace(range_[0], range_[1], length)

    data = np.zeros((values.shape[0], length))

    for i in range(values.shape[0]):
        data[i, :] = function(x, values[i, 0], values[i, 1], values[i, 2])

    return data

def find_nearest(array, value, averaging_number):
    """
    function to find the index of the nearest value in the array

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
    function to rotate the image

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



def layout_fig(graph, mod=None,x=1,y=1):

    """
    function

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
                             figsize=(3 * mod*x, y*3 * (graph // mod + (graph % mod > 0))))

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

    :param data: data need to be showed in image format
    :type data: array
    :param image: the output shape of the image
    :type image: array
    :param colorbar_shown: whether to show the color bar on the left of image
    :type colorbar_shown: boolean
    :param c_lim: Sets the scales of colorbar
    :type c_lim: list
    :param mod: set the number of image for each line
    :type mod: int
    :param title: set the title of figure
    :type title: string
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
        if colorbar_shown == True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax, format='%.1f')

            # Sets the scales
            if c_lim is not None:
                im.set_clim(c_lim)

    if title is not None:
        # Adds title to the figure
        fig.suptitle(title, fontsize=16,
                     y=1, horizontalalignment='center')

    fig.tight_layout()

    return fig

class global_scaler:



    def fit(self, data):

    # calculate the mean and standard deviation of the input array
        self.mean = np.mean(data.reshape(-1))
        self.std = np.std(data.reshape(-1))

    def fit_transform(self, data):
        """

        :param data: the input array
        :type data: array
        :return: the data get through the normalization
        :rtype: array
        """
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        """

        :param data: the input data
        :type: array
        :return: the data get through the normalization
        :rtype: array
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """

        :param data: the normalized array
        :type: array
        :return: the same scale of the raw data
        :rtype: array
        """
        return (data * self.std) + self.mean

printing = {'PNG':True,
            'EPS':False,
           'dpi': 300}
class generator:
    def __init__(self,
                 model,
                 scaled_data,
                 image,
                 channels=None,
                 color_map='viridis'):
        self.model = model
        self.image = image
        # defines the colorlist
        self.cmap = plt.get_cmap(color_map)
        self.modified_model = None

        if isinstance(model, type(DictionaryLearning())):
            def predictor(values):
                return np.dot(values, model.components_)

            self.predict = predictor
            self.vector_length = scaled_data.shape[1]
            self.embeddings = model.transform(scaled_data)
        elif np.atleast_3d(scaled_data).shape[2] == 1:

            def predictor(values):
                return model.decoder_model.predict(np.atleast_2d(values))

            self.embeddings = model.encoder_model.predict(np.atleast_3d(scaled_data))
            self.predict = predictor
            self.vector_length = scaled_data.shape[1]

        elif np.atleast_3d(scaled_data).shape[2] == 2:
            self.modified_model = 1

            def predictor(means, stds):

                return model.decoder_model.predict([np.atleast_2d(means), np.atleast_2d(stds)])

            self.emb_, self.mean, self.std = model.encoder_model.predict(np.atleast_3d(scaled_data))
            self.embeddings_tf = Sampling()([self.mean, self.std])
            self.embeddings = self.embeddings_tf.numpy()
            self.predict = predictor
            self.vector_length = scaled_data.shape[1]
        else:
            raise Exception('The model is not an included model type ')

        if channels == None:
            self.channels = range(self.embeddings.shape[1])
        else:
            self.channels = channels

    def generator_images(self,
                         folder,
                         ranges=None,
                         number_of_loops=200,
                         averaging_number=100,
                         graph_layout=[3, 3],
                         model_tpye = 'dog',
                         y_lim=[-2, 2],
                         y_lim_1 = [-2,2],
                         xlabel='Voltage (V)',
                         ylabel='',
                         xvalues=None
                         ):
        """

        :param folder: folder where to save
        :type folder: string
        :param ranges: range of the each embedding value
        :type ranges: list
        :param number_of_loops: embedding range divided by step size of it
        :type number_of_loops: int
        :param averaging_number: number of index which is nearest to the current value
        :type averaging_number: int
        :param graph_layout: format of output graph
        :type graph_layout: list
        :param y_lim: set the y scale
        :type y_lim: list
        :param xlabel: set the label of x axis
        :type xlabel; string
        :param ylabel: set the label of y axis
        :type ylabel: string
        :param xvalues: set the x axis
        :type xvalues: array

        """
        folder = make_folder(folder)
        for i in tqdm(range(number_of_loops)):
            # builds the figure
            # fig, ax = plt.subplots(graph_layout[0] // graph_layout[1] + (graph_layout[0] % graph_layout[1] > 0), graph_layout[1],
            #                       figsize=(3 * graph_layout[1], 3 * (graph_layout[0] // graph_layout[1] + (graph_layout[0] % graph_layout[1] > 0))))
            if model_tpye == 'dl':
                fig, ax = layout_fig(graph_layout[0] * 3, mod=graph_layout[1])
            else:
                fig, ax = layout_fig(graph_layout[0] * 4, mod=graph_layout[1])
            ax = ax.reshape(-1)
            # loops around all of the embeddings
            for j, channel in enumerate(self.channels):

                # checks if the value is None and if so skips tp next iteration
                if i is None:
                    continue

                if xvalues is None:
                    xvalues = range(self.vector_length)

                if ranges is None:
                    ranges = np.stack((np.min(self.embeddings, axis=0),
                                       np.max(self.embeddings, axis=0)), axis=1)

                # linear space values for the embeddings
                value = np.linspace(ranges[channel][0], ranges[channel][1],
                                    number_of_loops)
                # finds the nearest point to the value and then takes the average
                # average number of points based on the averaging number
                idx = find_nearest(
                    self.embeddings[:, channel],
                    value[i],
                    averaging_number)
                # computes the mean of the selected index

                if self.modified_model != None:
                    gen_mean = np.mean(self.mean[idx], axis=0)
                    gen_std = np.mean(self.std[idx], axis=0)

                    mn_ranges = np.stack((np.min(self.mean, axis=0),
                                          np.max(self.mean, axis=0)), axis=1)
                    sd_ranges = np.stack((np.min(self.std, axis=0),
                                          np.max(self.std, axis=0)), axis=1)

                    mn_value = np.linspace(mn_ranges[channel][0], mn_ranges[channel][1],
                                           number_of_loops)

                    sd_value = np.linspace(sd_ranges[channel][0], sd_ranges[channel][1],
                                           number_of_loops)

                    gen_mean[channel] = mn_value[i]

                    gen_std[channel] = sd_value[i]
                    generated = self.predict(gen_mean, gen_std).squeeze()

                if self.modified_model == None:

                    gen_value = np.mean(self.embeddings[idx], axis=0)
                    # specifically updates the value of the embedding to visualize based on the
                    # linear spaced vector
                    gen_value[channel] = value[i]
                    # generates the loop based on the model
                    generated = self.predict(gen_value).squeeze()
                # plots the graph



                # image_,angle_ = rotate_and_crop(self.embeddings[:, channel].reshape(self.image.shape[0:2]))
                ax[j].imshow(self.embeddings[:, channel].reshape(self.image.shape[0:2]), clim=ranges[channel])
                #                ax[j].imshow(image_, )
                ax[j].set_yticklabels('')
                ax[j].set_xticklabels('')
                y_axis,x_axis = np.histogram(self.embeddings[:,channel],number_of_loops)
                if model_tpye=='dog':
                    ax[j + len(self.channels)].plot(xvalues, generated,
                                                    color=self.cmap((i + 1) / number_of_loops))
                    # formats the graph
                    ax[j + len(self.channels)].set_ylim(y_lim[0], y_lim[1])
                    #   ax[j+len(self.channels)].set_yticklabels('Piezoresponse (Arb. U.)')
                    ax[j + len(self.channels)].set_ylabel('Amplitude of Spectural')
                    ax[j + len(self.channels)].set_xlabel(xlabel)
                    ax[j + len(self.channels) * 2].hist(self.embeddings[:,channel],number_of_loops)
                    ax[j + len(self.channels) * 2].plot(x_axis[i],y_axis[i],'ro')
                    ax[j + len(self.channels) * 2].set_ylabel('Distribution of Intensity')
                    ax[j + len(self.channels) * 2].set_xlabel('Range of Intensity')
                else:
                    if len(generated.shape)==1:
                        new_range = int(len(generated)/2)
                        generated_1 = generated[:new_range].reshape(new_range,1)
                        generated_2 = generated[new_range:].reshape(new_range,1)
                        generated = np.concatenate((generated_1,generated_2),axis=1)
                        if len(xvalues) !=  generated.shape[0]:
                            xvalues = range(int(self.vector_length / 2))



                    ax[j + len(self.channels)].plot(xvalues, generated[:, 0]*7.859902800847493e-05 -1.0487273116670697e-05
                                                    , color=self.cmap((i + 1) / number_of_loops))
                    # formats the graph
                    ax[j + len(self.channels)].set_ylim(y_lim[0], y_lim[1])
                    #   ax[j+len(self.channels)].set_yticklabels('Piezoresponse (Arb. U.)')
                    ax[j + len(self.channels)].set_ylabel('Piezoresponse (Arb. U.)')
                    ax[j + len(self.channels)].set_xlabel(xlabel)

                    ax[j + len(self.channels) * 2].plot(xvalues, generated[:, 1]*3.1454182388943095+1324.800141637855,
                                                        color=self.cmap((i + 1) / number_of_loops))
                    # formats the graph
                    ax[j + len(self.channels) * 2].set_ylim(y_lim_1[0], y_lim_1[1])
                    #      ax[j+len(self.channels)*2].set_yticklabels('Resonance (KHz)')
                    ax[j + len(self.channels) * 2].set_ylabel('Resonance (KHz)')
                    ax[j + len(self.channels) * 2].set_xlabel(xlabel)
                    ax[j + len(self.channels) * 3].hist(self.embeddings[:, channel], number_of_loops)
                    ax[j + len(self.channels) * 3].plot(x_axis[i], y_axis[i], 'ro')
                    ax[j + len(self.channels) * 3].set_ylabel('Distribution of Intensity')
                    ax[j + len(self.channels) * 3].set_xlabel('Range of Intensity')

                # gets the position of the axis on the figure
                # pos = ax[j].get_position()
                # plots and formats the binary cluster map
                # axes_in = plt.axes([pos.x0 - .03, pos.y0, .06 , .06])
                ## rotates the figure
                # if plot_format['rotation']:
                #    imageb, scalefactor = rotate_and_crop(embeddings[:, j].reshape(image.shape),
                #                                          angle=plot_format['angle'], frac_rm=plot_format['frac_rm'])
                # else:
                #    scalefactor = 1
                #    imageb = encode_small[:, j].reshape(60, 60)
                # plots the imagemap and formats
                # image_,angle_ = rotate_and_crop()

            ax[0].set_ylabel(ylabel)
            fig.tight_layout()
            savefig(pjoin(folder, f'{i:04d}_maps'), printing)

            plt.close(fig)

def embedding_maps_movie(data, image, printing, folder, beta,loss,
                   filename='./embedding_maps', c_lim=None,mod=4,colorbar_shown=True):
    """
    plots the embedding maps from a neural network

    Parameters
    ----------
    data : raw data to plot of embeddings
        data of embeddings
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    plot_format  : dict
        sets the plot format for the images
    folder : string
        set the folder where to export the images
    verbose : bool (optional)
        sets if the code should report information
    letter_labels : bool (optional)
        sets is labels should be included
    filename : string (optional)
        sets the filename for saving
    num_of_plots : int, optional
            number of principal components to show
    ranges : float, optional
            sets the clim of the images

    return
    ----------

    fig : object
        the figure pointer
    """


    # creates the figures and axes in a pretty way
    fig, ax = layout_fig(data.shape[1], mod)
    title_name = 'beta='+beta+'_loss='+loss
    fig.suptitle(title_name,fontsize=12,y=1)
    for i, ax in enumerate(ax):
        if i < data.shape[1]:
            im = ax.imshow(data[:, i].reshape(image.shape[0], image.shape[1]))
            ax.set_xticklabels('')
            ax.set_yticklabels('')

            # adds the colorbar
        if colorbar_shown == True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax, format='%.1e')

            # Sets the scales
            if c_lim is not None:
                im.set_clim(c_lim)

    # plots all of the images



    plt.tight_layout(pad=1)
    fig.set_size_inches(12, 12)
    # saves the figure
    fig.savefig(folder + '/' + filename +'.png', dpi=300)
#    savefig(folder + '/' + filename, printing)

    return(fig)


def training_images(model,
                    data,
                    image,
                    number_layers,
                    model_folder,
                    beta,
                    printing,
                    folder,
                    file_name):
    """
    plots the training images

    Parameters
    ----------
    model : tensorflow object
        neural network model
    data : float, array
        sets the line graph to plot
    model_folder : float, array
        sets the embedding map to plot
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    plot_format  : dict
        sets the plot format for the images
    folder : string
        set the folder where to export the images
    data_type : string (optional)
        sets the type of data which is used to construct the filename

    """

    # makes a copy of the format information to modify
    printing_ = printing.copy()


    # sets to remove the color bars and not to print EPS

    printing_['EPS'] = False

    # simple function to help extract the filename
    def name_extraction(filename):
        filename = file_list[0].split('/')[-1][:-5]
        return filename

    embedding_exported = {}

    # searches the folder and finds the files
    file_list = glob.glob(model_folder + '/phase_shift_only*')
    file_list = natsorted(file_list, key=lambda y: y.lower())

    for i, file_list in enumerate(file_list):
        # load beta and loss value
        loss_ = file_list[-12:-5]

        # loads the weights into the model
        model.load_weights(file_list)

        # Computes the low dimensional layer
        embedding_exported[name_extraction(file_list)] = get_activations(model, data, number_layers)

        # plots the embedding maps
        _ = embedding_maps_movie(embedding_exported[name_extraction(file_list)], image,printing_,
                           folder, beta, loss_, filename='./' + file_name + '_epoch_{0:04}'.format(i))

        # Closes the figure
        plt.close(_)

