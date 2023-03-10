import os
import numpy as np
from astropy.io import fits
import random
import copy
import  matplotlib.pyplot as plt

# For multidim sampling maps/grid, if not conserve pdf simply minimize, else minimize without replacement and start by the most emissive pixels
# Conserve PDF = copying cloud. Else = use sample_cloud as line library
# Add replacement option

def compute_noise(cube, vmin, vmax):
    """

    :param cube:
    :param vmin:
    :param vmax:
    :return:
    """
    noise_cube = np.concatenate((cube[0:vmin,:,:],cube[vmax:,:,:]), axis = 0)
    noise = np.nanstd(noise_cube, axis = 0) #p.sqrt(np.nanmean(noise_cube**2, axis = 0))
    return noise


def channelShiftVec(x, ChanShift):
    """

    :param x:
    :param ChanShift:
    :return:
    """

    # Routine from E. Rosolowsky
    # Shift an array of spectra (x) by a set number of Channels (array)
    ftx = np.fft.fft(x, axis=0)
    m = np.fft.fftfreq(x.shape[0])
    phase = np.exp(-2 * np.pi * m[:, np.newaxis]
                   * 1j * ChanShift[np.newaxis, :])
    x2 = np.real(np.fft.ifft(ftx * phase, axis=0))
    return(x2)

def ShuffleCube(DataCube, centroid_map, chunk=1000):
    """

    :param DataCube:
    :param centroid_map:
    :param chunk:
    :return:
    """

    # Routine from E. Rosolowsky
    spaxis = np.linspace(0, DataCube.shape[0], DataCube.shape[0] - 1)#DataCube.spectral_axis
    y,x = np.where(np.isfinite(centroid_map))
    relative_channel = np.arange(len(spaxis)) - (len(spaxis) // 2)
    centroids = centroid_map[y, x]
    sortindex = np.argsort(spaxis)
    channel_shift = -1 * np.interp(centroids, spaxis[sortindex],
                                   np.array(relative_channel[sortindex], dtype=float))
    NewCube = np.empty(DataCube.shape)
    NewCube.fill(np.nan)
    nchunk = (len(x) // chunk)
    for thisx, thisy, thisshift in zip(np.array_split(x, nchunk),
                                       np.array_split(y, nchunk),
                                       np.array_split(channel_shift, nchunk)):
        spectrum = DataCube[:, thisy, thisx]
        baddata = ~np.isfinite(spectrum)
        shifted_spectrum = channelShiftVec(np.nan_to_num(spectrum),
                                           np.atleast_1d(thisshift))
        shifted_spectrum[baddata>0] = np.nan
        NewCube[:, thisy, thisx] = shifted_spectrum

    return NewCube

def standardize(img):
    """

    :param img:
    :return:
    """
    img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
    return img

def generate_grid(coord, img_size):
    """

    :param coord:
    :param img_size:
    :return:
    """
    x = np.linspace(0, img_size[0], img_size[0])
    y = np.linspace(0, img_size[1], img_size[1])
    X,Y = np.meshgrid(x,y)
    xc, yc = coord
    r = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    return r


"""                                                                                   ""
########################################################################################
########################################################################################
"""                                                                                   ""

class CloudGenerator:
    def __init__(self,
                 line,
                 size,
                 save_directory = os.getcwd(),
                 conserve_pdf = True):

        self.gen_map = None
        self.sample_cube = None
        self.line = line
        self.morphology = None# gen_map.morphology
        self.size = size
        self.save_directory = save_directory
        self.conserve_pdf = conserve_pdf
        #self.sample_map = sample_map



    def gen_cube(self, samples, sample_map, gen_map, conserve_pdf = True, s2n = 0, vmin = 20, vmax = -20, limits = None):
        """

        :param sample_cube:
        :param sample_map:
        :param gen_map:
        :param s2n:
        :param conserve_pdf:
        :return:
        """
        self.morphology = gen_map.morphology
        #gen_map = standardize(gen_map)
        #sample_map = standardize(sample_map)

        if conserve_pdf == True:
            gen_cube = samples#Cube(self.line, samples, save_directory=self.save_directory)#copy.deepcopy(cube_sample)
            gen_cube.save_directory = self.save_directory
            ids_gen = np.argsort(gen_map.data.reshape(-1))
            ids_sample = np.argsort(sample_map.reshape(-1)) #np.argsort(np.nansum(gen_cube.data, axis = 0).reshape(-1))
            gen_cube.data = np.reshape(gen_cube.data,(gen_cube.spec_len,-1))
            gen_cube.data[:,ids_gen] = gen_cube.data[:,ids_sample]
            gen_cube.data = np.reshape(gen_cube.data,(gen_cube.spec_len,gen_cube.size[0], gen_cube.size[1]))

        elif conserve_pdf == False:
            print('to add')
            gen_cube = Cube(self.line, samples, save_directory=self.save_directory)#copy.deepcopy(cube_sample)

        else: # Catch error
            print('error')
            #gen_cube = Cube(self.line, samples, save_directory=self.save_directory)#copy.deepcopy(cube_sample)

        #self.cube_gen = gen_cube
        gen_cube.morphology = gen_map.morphology
        gen_cube.gen_params = gen_map.gen_params
        gen_cube.save_directory = self.save_directory

        return gen_cube

"""
Cube Class
"""
class Cube: # Ultimately will require a header variable for saving
    def __init__(self,
                 line = '12co10',
                 data = np.zeros(shape=(3,3,3)),
                 save_directory = os.getcwd(),
                 vel_res = 1):

        self.morphology = None
        self.data = data
        self.line = line
        self.size = data.shape[1::]
        self.spec_len = data.shape[0]
        self.gen_params = None
        self.save_directory = save_directory
        self.vel_res = vel_res


    def compute_moment(self, order, vmin = 20, vmax = -20):
        """

        :param order:
        :param vmin:
        :param vmax:
        :return:
        """
        if type(order) == int:
            velocity = np.arange(0.5,(self.spec_len+1)*0.5, 0.5)
            velocity = np.broadcast_to(velocity, (self.size[1],self.size[0], len(velocity)))
            velocity = velocity.T

            data = self.data * self.vel_res


            M0 = np.nansum(data, axis = (0))

            if order == 0:
                return  M0

            M1 = np.nansum(data * velocity, axis = (0))  / M0
            if order == 1:
                return M1

            if order > 1:
                moment = np.nansum(data*(velocity - M1) ** order , axis = 0)  / M0
                return moment

        if order == 'tpeak':
            return np.nanmax(self.data, axis = 0)

        if order == 'noise':
            return compute_noise(self.data, vmin, vmax)

        if order == 'vpeak':
            return np.nanargmax(self.data, axis = 0)

    def get_sample(self, size, s2n = 0, vmin = 20, vmax = -20, limits = None, coords_sample = False): #74 160 #Bug if generated map marger than sample cube with limits
        """

        :param size:
        :param s2n:
        :param vmin:
        :param vmax:
        :param limits:
        :return:
        """
        bad_ids = np.nanmax(self.data, axis = 0) > s2n * compute_noise(self.data, vmin, vmax)
        map_img_line = np.sum(self.data, axis = 0)
        map_img_line[~bad_ids] = np.nan
        map_img_line = (map_img_line).reshape((-1))
        p = np.ones(map_img_line.shape)
        coords = np.arange(0, map_img_line.shape[0], 1)
        p[np.isnan(map_img_line)] = 0
        p[~np.isnan(map_img_line)] = 1 / len(p[~np.isnan(map_img_line)])

        if type(coords_sample) == bool:
            coords_sample = np.random.choice(coords, size = size[0] * size[1], replace = True, p = p)

        cube_dat = copy.deepcopy(self.data)
        if limits:
            cube_dat = cube_dat[:, limits[0]:limits[1], limits[2]:limits[3]]
        cube_dat = cube_dat.reshape((self.spec_len,-1))
        cube_sample = cube_dat[:,coords_sample]
        cube_sample = cube_sample.reshape((self.spec_len,size[0], size[1]))

        return Cube(line = self.line, data = cube_sample, save_directory=self.save_directory, vel_res=self.vel_res), coords_sample

    def gen_vel_gradient(self,  amplitude, angle, offset = 0):
        """

        :param center_vel:
        :param amplitude:
        :param angle:
        :param img_size:
        :return:
        """

        center_vel = int(self.spec_len/2)
        x = np.arange(0, self.size[0], 1)
        y = np.arange(0, self.size[1], 1)
        X, Y = np.meshgrid(x, y)

        angle = angle * np.pi / 180
        max_len = np.nanmax(self.size) * (np.cos(angle) + np.sin(angle))
        a = amplitude / max_len
        b = - a / 2 * max_len

        gradient = (a * (X * np.cos(angle) + Y * np.sin(angle)) + b) + center_vel + offset

        return gradient

    def apply_vel_gradient(self, gradient):
        """

        :param gradient:
        :return:
        """
        self.data = ShuffleCube(self.data, gradient)


    def get_filename(self):
        """

        :return:
        """
        filename = self.save_directory + '/' +  self.morphology + '/' + self.line + '_' + self.morphology + '_'
        if not os.path.exists(self.save_directory + '/' +  self.morphology):
            os.mkdir(self.save_directory + '/' +  self.morphology)
        for param in self.gen_params:
            filename +=  str(param) + '_'
        filename += '.fits'
        return filename

    def save_cloud(self):
        """

        :return:
        """
        fn = self.get_filename()
        hdu = fits.PrimaryHDU(self.data.astype('float32'))
        hdu.writeto(fn, overwrite=True)

    def toSpectralCube(self):
        print('to implement')
        return None



"""
Grid Class
"""
class MapSample: #Make a single grid class for sample or generated, will require new methods when physical parameters are added
    def __init__(self, data, moment):
        self.data = data
        self.size = np.shape(data)
        self.moment = moment


"""
GenMap Class
"""
class GenMap:
    def __init__(self,  size, seed ):
        self.size = size
        #self.dimension = dimension
        self.morphology = ''
        self.data = np.zeros(shape=(size[0], size[1])) #np.zeros(shape=[dimension, size[0], size[1]])
        self.gen_params = []
        self.seed = seed


    def generate_fBm(self,beta):
        """

        :param n:
        :param m:
        :param beta:
        :return:
        """
        n = self.size[0]
        m = self.size[1]
        x = np.arange(0, n, 1)
        y = np.arange(0, m, 1)
        X, Y = np.meshgrid(x, y)
        xc, yc = int(n/2), int(m/2)

        freqs = np.sqrt((X-xc)**2 + (Y-yc)**2)
        amplitudes = 1/(np.power(freqs,(beta/2))) #
        np.random.seed(self.seed)
        phases = np.random.uniform(0, np.pi*2, size = (n,m))

        amplitudes[np.isinf(amplitudes)] = 1e8
        amplitudes = np.maximum( amplitudes, amplitudes.T) # Make Fourier 2D space symmetrical
        amplitudes = amplitudes * np.exp(1j * phases)

        amplitudes = np.fft.ifftshift(amplitudes)
        Z = np.fft.ifft2(amplitudes)

        Z = (abs(Z))

        self.data = Z
        self.morphology = 'fBm'
        self.gen_params = [beta, self.seed]

        #return seed


    def generate_gaussian(self, R, a, sigma, coord, img_size):
        """

        :param a: #unit cm-3
        :param sigma: #unit pc
        :param img_size:
        :param coord:
        :return:
        """
        R = np.sqrt(R**2)
        r = generate_grid(coord,img_size)
        gaussian = lambda a,r,R,sigma : a * np.exp(-(r-R)**2 / (2 * sigma**2))
        Z = gaussian(a,r,R,sigma)
        Z[np.isnan(Z)] = 0
        self.data = Z
        self.morphology = 'Gaussian'
        self.gen_params = [R,a,sigma]


        #return Z

    def generate_torus(self, R,r,coord, img_size):
        """

        :param R:
        :param r:
        :param coord:
        :param img_size:
        :return:
        """
        rx = generate_grid(coord,img_size)
        torus = lambda r,R,rx : np.sqrt(-(rx - R)**2 + r**2)
        Z = torus(r,R,rx)
        Z[np.isnan(Z)] = 0
        self.data = Z
        self.morphology = 'Torus'
        self.gen_params = [R,r]

        #return Z

    def generate_density_power_law(self, R, n0,a, coord, img_size):
        """

        :param a: #2.5, 1.6
        :param coord:
        :param img_size:
        :return:
        """
        R = np.sqrt(R**2)
        r = generate_grid(coord,img_size)
        XY = abs(r-R)+1
        pl = lambda n0, XY, a : n0 * np.power(XY,-a)
        Z = pl(n0, XY, a)
        Z[np.isnan(Z)] = 0
        self.data = Z
        self.morphology = 'Density_pl'
        self.gen_params = [R,n0, a]

        #return Z

    def generate_plummer(self, R, n0, Rflat, p ,coord, img_size):
        """

        :param n0: #cm-3
        :param Rflat: #pc
        :param p: # no unit
        :param coord:
        :param img_size:
        :return:
        """
        R = np.sqrt(R**2)
        r = generate_grid(coord,img_size)
        plummer = lambda n0,r,R,Rflat,p : n0 / np.power(( 1 + ((r-R)/Rflat)**2 ),p/2)
        Z = plummer(n0,r,R,Rflat,p)
        Z[np.isnan(Z)] = 0
        self.data = Z
        self.morphology = 'Plummer'
        self.gen_params = [R, n0, Rflat, p]

        #return Z
