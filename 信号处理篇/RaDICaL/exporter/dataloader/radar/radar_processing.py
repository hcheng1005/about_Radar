import numpy as np
from mmwave import dsp

class RadarFrameV1(object):
    """docstring for RadarFrame"""
    def __init__(self, radar_config,
                 angle_res = 1,
                 angle_range = 90,
                 origin_at_bottom_center = True,
                 use_float32 = False,
                ):
        super(RadarFrameV1, self).__init__()
        self.cfg = radar_config

        #Beamforming params
        self.bins_processed = self.cfg['profiles'][0]['adcSamples'] #radar_cube.shape[0]
        self.virt_ant = self.cfg['numLanes'] * len(self.cfg['chirps']) #radar_cube.shape[1]
        self.doppler_bins = self.cfg['numChirps'] // len(self.cfg['chirps']) #radar_cube.shape[2]
        self.angle_res = angle_res
        self.angle_range = angle_range
        self.angle_bins = (self.angle_range * 2) // self.angle_res + 1
        self.num_vec, self.steering_vec = dsp.gen_steering_vec(self.angle_range, 
                                                               self.angle_res, 
                                                               self.virt_ant)

        #Properties
        self.__range_azimuth_dirty = True
        self.__range_azimuth = np.zeros((self.bins_processed, self.angle_bins), 
                                        dtype=np.complex64 if use_float32 else np.complex_)
        self.__beam_weights = np.zeros((self.virt_ant, self.bins_processed), 
                                       dtype=np.complex64 if use_float32 else np.complex_)
        self.__range_doppler = None
        self.__raw_cube = None
        self.__range_cube = None

        self.__flip_ra = origin_at_bottom_center

    @property
    def max_range(self):
        return self.range_resolution * self.bins_processed

    @property
    def range_resolution(self):
        range_res, bw = dsp.range_resolution(self.cfg['profiles'][0]['adcSamples'],
                                             self.cfg['profiles'][0]['adcSampleRate'] / 1000,
                                             self.cfg['profiles'][0]['freqSlopeConst'] / 1e12)
        return range_res

    @property
    def doppler_resolution(self):
        _, bw = dsp.range_resolution(self.cfg['profiles'][0]['adcSamples'],
                                             self.cfg['profiles'][0]['adcSampleRate'] / 1000,
                                             self.cfg['profiles'][0]['freqSlopeConst'] / 1e12)
        return dsp.doppler_resolution(bw,
                                      start_freq_const=self.cfg['profiles'][0]['start_frequency'] / 1e9,
                                      ramp_end_time=self.cfg['profiles'][0]['rampEndTime'] * 1e6,
                                      idle_time_const=self.cfg['profiles'][0]['idle'] * 1e6,
                                      num_loops_per_frame=self.cfg['numChirps'] / len(self.cfg['chirps']),
                                      num_tx_antennas=self.cfg['numTx'])


    @property
    def max_unambiguous_doppler(self):
        raise NotImplementedError

    @property
    def raw_cube(self):
        return self.__raw_cube

    @raw_cube.setter
    def raw_cube(self, raw_cube):
        self.__raw_cube = raw_cube
        self.__range_cube = None
        self.__range_doppler = None
        self.__range_azimuth_dirty = True

    @property
    def range_cube(self):
        if self.__range_cube:
            return self.__range_cube
        else:
            range_cube = dsp.range_processing(self.raw_cube)
            self.__range_cube = np.swapaxes(range_cube, 0, 2)
            return self.__range_cube

    @property
    def range_doppler(self):
        if self.__range_doppler:
            return self.__range_doppler
        else:
            range_doppler = dsp.doppler_processing(self.raw_cube)
            self.__range_doppler = range_doppler
            return self.__range_doppler

    @property
    def range_azimuth_capon(self):
        if not self.__range_azimuth_dirty:
            r = self.__range_azimuth
        else:
            self.__aoa_capon_process()
            r = self.__range_azimuth

        if self.__flip_ra:
            return np.flipud(np.fliplr(r))
        else:
            return r

    def __aoa_capon_process(self):
        radar_cube = self.range_cube

        for jj in range(self.bins_processed):
            self.__range_azimuth[jj,:], self.__beam_weights[:,jj] = dsp.aoa_capon(radar_cube[jj],
                                                                      self.steering_vec)

        self.__range_azimuth_dirty = False

    def compute_range_azimuth(self, radar_raw=None):
        if radar_raw is not None:
            self.raw_cube = radar_raw

        return self.range_azimuth_capon
