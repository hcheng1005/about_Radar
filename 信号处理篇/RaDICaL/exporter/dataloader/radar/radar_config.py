def read_radar_params(filename):
    with open(filename) as cfg:
        iwr_cmds = cfg.readlines()
        iwr_cmds = [x.strip() for x in iwr_cmds]
        radar_cfg = cfg_list_to_dict(iwr_cmds)

    print(radar_cfg)
    return radar_cfg


def cfg_list_to_dict(data):
    # make sure the order of the headers and functions match
    cfg_headers = ['channelCfg','profileCfg','chirpCfg','frameCfg','adcCfg','lowPower']
    cfg_funcs = [channelStr_to_dict, profileStr_to_dict, chirp_to_dict,frameStr_to_dict,adcStr_to_dict,power_to_dict]
    cfg = {}
    for element in data:
        for header,func in zip(cfg_headers,cfg_funcs):
            if header in element:
                line = element.split()[1:]  # Gets info between quotes
                cfg = func(line, cfg)
    return cfg


def dict_to_list(cfg):
    cfg_list = ['flushCfg','dfeDataOutputMode 1']


    # rx antennas/lanes for channel config
    rx_bool = [cfg['rx4'], cfg['rx3'], cfg['rx2'], cfg['rx1']]
    rx_mask = sum(2 ** i for i, v in enumerate(reversed(rx_bool)) if v)
    # number of tx antennas for channel config
    tx_bool = [cfg['tx3'], cfg['tx2'], cfg['tx1']]
    tx_mask = sum(2 ** i for i, v in enumerate(reversed(tx_bool)) if v)
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) is False else 0
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is False and (cfg['tx1'] or cfg['tx3']) is True else 0
    #print('[NOTE] Elevation and Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) else 0
    cfg_list.append('channelCfg %s %s 0' % (rx_mask, tx_mask))  # rx and tx mask

    # adc config
    if cfg['isComplex'] and cfg['image_band']:
        outputFmt = 2
        #print('[NOTE] Complex 2x mode, both Imaginary and Real IF spectrum is filtered and sent to ADC, so\n'
        #      '       if Sampling rate is X, ADC data would include frequency spectrum from -X/2 to X/2.')
    elif cfg['isComplex'] and not cfg['image_band'] == True:
        outputFmt = 1
        #print('[NOTE] Complex 1x mode, Only Real IF Spectrum is filtered and sent to ADC, so if Sampling rate\n'
        #      '       is X, ADC data would include frequency spectrum from 0 to X.')
    else: raise ValueError("Real Data Type Not Supported")
    cfg_list.append('adcCfg 2 %s' % outputFmt)  # 16 bits (mandatory), complex 1x or 2x

    # adc power
    if cfg['adcPower'] =='low':
        power_mode = 1
        #print('[NOTE] The Low power ADC mode limits the sampling rate to half the max value.')
    elif cfg['adcPower'] =='regular': power_mode = 0
    else: raise ValueError("ADC power level Not Supported")
    cfg_list.append('lowPower 0 %s' % power_mode)  # power mode

    # profile configs
    for profile_ii in cfg['profiles']:
        cfg_list.append('profileCfg %s %s %s %s %s %s %s %s %s %s %s %s %s %s'
                % (profile_ii['id'],
                float(profile_ii['start_frequency']/1e9),
                float(profile_ii['idle']/1e-6),
                float(profile_ii['adcStartTime']/1e-6),
                float(profile_ii['rampEndTime']/1e-6),
                int(profile_ii['txPower']),
                int(profile_ii['txPhaseShift']),
                float(profile_ii['freqSlopeConst']/1e12),
                float(profile_ii['txStartTime']/1e-6),
                int(profile_ii['adcSamples']),
                int(profile_ii['adcSampleRate']/1e3),
                int(profile_ii['hpfCornerFreq1']),
                int(profile_ii['hpfCornerFreq2']),
                int(profile_ii['rxGain'])))

    # chirp configs
    for chirp_ii in cfg['chirps']:

        # Check if chirp is referring to valid profile config
        profile_valid = False
        for profile_ii in cfg['profiles']:
            if chirp_ii['profileID'] == profile_ii['id']: profile_valid = True
        if profile_valid is False: raise ValueError("The following profile id used in chirp "
                                                    "is invalid: %i" % chirp_ii['profileID'])
        ###############################################################################################################
        '''
        # check if tx values are valid
        if hamming([chirp_ii['chirptx3'],chirp_ii['chirptx2'],chirp_ii['chirptx1']],
            [cfg['tx3'], cfg['tx2'], cfg['tx1']])*3 > 1:
            raise ValueError("Chirp should have at most one different Tx than channel cfg")
        '''
        ###############################################################################################################
        if chirp_ii['chirpStartIndex'] > chirp_ii['chirpStopIndex']: raise ValueError("Particular chirp start index after chirp stop index")
        tx_bool = [chirp_ii['chirptx3'],chirp_ii['chirptx2'],chirp_ii['chirptx1']]
        tx_mask = sum(2 ** i for i, v in enumerate(reversed(tx_bool)) if v)
        cfg_list.append('chirpCfg %s %s %s %s %s %s %s %s'
                % (chirp_ii['chirpStartIndex'],
                   chirp_ii['chirpStopIndex'],
                   chirp_ii['profileID'],
                   chirp_ii['startFreqVariation'],
                   chirp_ii['slopeVariation'],
                   chirp_ii['idleVariation'],
                   chirp_ii['adcStartVariation'],
                   tx_mask))

    # frame config
    chirpStop = 0
    chirpStart = 511  # max value for chirp start index
    for chirp_ii in cfg['chirps']:
        chirpStop = max(chirpStop, chirp_ii['chirpStopIndex'])
        chirpStart = min(chirpStart,chirp_ii['chirpStartIndex'])
    chirps_len  = chirpStop + 1

    numLoops = cfg['numChirps']/chirps_len
    if chirpStart > chirpStop: raise ValueError("Chirp(s) start index is after chirp stop index")
    if numLoops % 1 != 0: raise ValueError("Number of loops is not integer")
    if numLoops > 255 or numLoops < 1: raise ValueError("Number of loops must be int in [1,255]")

    numFrames = cfg['numFrames'] if 'numFrames' in cfg.keys() else 0  # if zero => inf

    cfg_list.append('frameCfg %s %s %s %s %s 1 0'
            % (chirpStart, chirpStop, int(numLoops), numFrames, 1000/cfg['fps']))

    cfg_list.append('testFmkCfg 0 0 0 1')
    cfg_list.append('setProfileCfg disable ADC disable')
    return cfg_list


def channelStr_to_dict(args, curr_cfg=None):

    if curr_cfg:
        cfg = curr_cfg
    else:
        cfg = {}

    # This is the number of receivers which is equivalent to the number of lanes in the source code
    # Later, may include the result from the number of transmitters
    rx_bin = bin(int(args[0]))[2:].zfill(4)
    cfg['numLanes'] = len([ones for ones in rx_bin if ones == '1'])
    (cfg['rx4'],cfg['rx3'],cfg['rx2'],cfg['rx1']) = [bool(int(ones)) for ones in rx_bin]

    # This is the number of transmitters
    tx_bin = bin(int(args[1]))[2:].zfill(3)
    cfg['numTx'] = len([ones for ones in tx_bin if ones == '1'])
    (cfg['tx3'], cfg['tx2'], cfg['tx1']) = [bool(int(ones)) for ones in tx_bin]
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) is False else 0
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is False and (cfg['tx1'] or cfg['tx3']) is True else 0
    #print('[NOTE] Elevation and Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) else 0


    return cfg


def profileStr_to_dict(args, curr_cfg=None):
    normalizer = [None, 1e9, 1e-6, 1e-6, 1e-6, None, None, 1e12, 1e-6, None, 1e3, None, None, None]
    dtype = [int, float, float, float, float, float, float, float, float, int, float, int, int, float]
    keys = ['id',
            'start_frequency',
            'idle',
            'adcStartTime',
            'rampEndTime',
            'txPower',
            'txPhaseShift',
            'freqSlopeConst',
            'txStartTime',
            'adcSamples',
            'adcSampleRate',
            'hpfCornerFreq1',
            'hpfCornerFreq2',
            'rxGain',
            ]
    # Check if the main dictionary exists
    if curr_cfg:
        cfg = curr_cfg
        if 'profiles' not in cfg.keys():
            cfg['profiles']=[]
    else:
        cfg = {'profiles': []}

    profile_dict = {}
    for k, v, n, d in zip(keys, args, normalizer, dtype):
        profile_dict[k] = d(float(v) * n if n else v)

    cfg['profiles'].append(profile_dict)
    return cfg


def chirp_to_dict(args,curr_cfg=None):
    if curr_cfg:
        cfg = curr_cfg
        if 'chirps' not in cfg.keys():
            cfg['chirps'] = []
    else:
        cfg = {'chirps': []}

    chirp_dict = {}
    chirp_dict['chirpStartIndex'] = int(args[0])
    chirp_dict['chirpStopIndex'] = int(args[1])
    chirp_dict['profileID'] = int(args[2])
    chirp_dict['startFreqVariation'] = float(args[3])
    chirp_dict['slopeVariation'] = float(args[4])
    chirp_dict['idleVariation'] = float(args[5])
    chirp_dict['adcStartVariation'] = float(args[6])

    tx_bin = bin(int(args[7]))[2:].zfill(3)
    (chirp_dict['chirptx3'], chirp_dict['chirptx2'], chirp_dict['chirptx1']) = [bool(int(ones)) for ones in tx_bin]

    cfg['chirps'].append(chirp_dict)
    return cfg


def power_to_dict(args,curr_cfg=None):
    if curr_cfg:
        cfg = curr_cfg
    else:
        cfg = {}
    if int(args[1]) ==1:
        cfg['adcPower'] = 'low'
        #print('[NOTE] The Low power ADC mode limits the sampling rate to half the max value.')
    elif int(args[1]) ==0:
        cfg['adcPower'] = 'regular'
    else:
        raise ValueError ("Invalid Power Level")
    return cfg


def frameStr_to_dict(args, cfg):

    # Number of chirps
    if 'chirps' not in cfg.keys():
        raise ValueError("Need to define chirps before frame")

    chirpStop =0
    for ii in range(len(cfg['chirps'])):
        chirpStop = max(chirpStop,cfg['chirps'][ii]['chirpStopIndex'])
    chirps_len = chirpStop + 1

    cfg['numChirps'] = int(args[2]) * chirps_len  # num loops * len(chirps)
    if int(args[3]) != 0: cfg['numFrames'] = int(args[3])

    # args[4] is the time in milliseconds of each frame
    cfg['fps'] = 1000/float(args[4])


    return cfg


def adcStr_to_dict(args, curr_cfg=None):
    if curr_cfg:
        cfg = curr_cfg
    else:
        cfg = {}

    if int(args[1]) == 1:
        cfg['isComplex'] = True
        cfg['image_band'] = False
        #print('[NOTE] Complex 1x mode, Only Real IF Spectrum is filtered and sent to ADC, so if Sampling rate\n'
        #      '       is X, ADC data would include frequency spectrum from 0 to X.')
    elif int(args[1]) == 2:
        cfg['isComplex'] = True
        cfg['image_band'] = True
        #print('[NOTE] Complex 2x mode, both Imaginary and Real IF spectrum is filtered and sent to ADC, so\n'
        #      '       if Sampling rate is X, ADC data would include frequency spectrum from -X/2 to X/2.')
    else:
        raise ValueError("Real Data Type Not Supported")

    return cfg
