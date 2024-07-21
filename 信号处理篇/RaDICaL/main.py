import numpy as np  
import matplotlib.pyplot as plt  
from glob import glob
    
from tqdm import tqdm
from absl import app, flags, logging

import rosbag
import cv_bridge
import cv2

from exporter.dataloader.h5_dataset import H5DatasetSaver, H5UnalignedDatasetSaver
from exporter.dataloader.stream_alignment import SensorStreamAlignment

from exporter.dataloader.radar import read_radar_params, rosmsg_to_radarcube_v1
import OpenRadar.mmwave.dsp as dsp


def plot_range_doppler(plt, ax, range_fft_data, title):  
    """  
    Plot range-Doppler map.  
    """  
    magnitude = np.abs(range_fft_data)  
    log_magnitude = 20 * np.log10(magnitude + 1e-6)  # Convert to dB scale      
    # Update the radar plot  
    ax.clear()  
    ax.imshow(magnitude, aspect='auto',  origin='lower')  
    # Draw and pause to update the plots  
    plt.draw()  
    plt.pause(0.01)   
    
    

if __name__ == '__main__':
    # reada mmWave Radar cfg
    radar_cfg = read_radar_params('/home/charles/Code/RaDICal/demo/radarcfg/outdoor_human_rcs_50m.cfg')    
        
    # 打开rosbag文件  
    bag = rosbag.Bag('/mnt/hgfs/dataset/DOI-10-13012-b2idb-3289560_v1/DOI-10-13012-b2idb-3289560_v1/50m_collection/2020-07-25-10-48-29.bag')  

    save_counter = 0  
    SAVE_FLAG = 0
    DEBUG_FLAG = 1

    # Set up the plot  
    plt.rcParams['figure.figsize'] = [10, 10]  
    plt.ion()  
    fig, ax1 = plt.subplots()  
    
    TOPICS_MAP = {'radar':'/radar_data',
                'depth':'/camera/aligned_depth_to_color/image_raw',
                'rgb':'/camera/color/image_raw',
                'imu_accel':'/camera/accel/sample',
                'imu_gyro':'/camera/gyro/sample',
                }

    NAMES_MAP = { v:k for (k, v) in TOPICS_MAP.items() }
    all_topics = list(bag.get_type_and_topic_info().topics.keys())
    stream_names = list([NAMES_MAP[t] for t in all_topics if t in NAMES_MAP.keys()])

    topic_names = list([TOPICS_MAP[t] for t in stream_names])
    topic_to_stream_idx = { t: idx for (t, idx) in zip(topic_names, range(len(topic_names))) }

    bridge = cv_bridge.CvBridge()
    
    logging.info("Iterating through bag messages")
    for topic, msg, t in tqdm(bag.read_messages(topics=topic_names),
                              total=bag.get_message_count(topic_names)):
        if topic == '/camera/color/image_raw':
            # img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # something wrong 
            continue
        elif topic == '/radar_data':
            
            """
            Here is Radar DSP PROCESS
            """
            # 提取雷达数据  
            radar_cube_raw = rosmsg_to_radarcube_v1(msg, radar_cfg)
            
            radar_cube_raw = radar_cube_raw - np.mean(radar_cube_raw, axis=-1, keepdims=True)
            
            # do rangeFFT first
            radar_cube = dsp.range_processing(radar_cube_raw, window_type_1d=dsp.utils.Window.HANNING) # 4 is HANNING
            
            # then do dopplerFFT first
            fft2d_log_abs, aoa_input = dsp.doppler_processing(radar_cube,interleaved=False)
            
            # and do fftshift
            fft2d_log_abs = np.fft.fftshift(fft2d_log_abs, axes=-1)
            
            # and do CFAR
            peaks = dsp.cfar2d_ca(fft2d_log_abs,debug=DEBUG_FLAG)
            if DEBUG_FLAG:
                print(np.where(peaks == 1))
            
            # 可视化
            ax1.clear()  
            ax1.imshow(fft2d_log_abs, aspect='auto',  origin='lower')  
            plt.scatter(np.where(peaks == 1)[1], np.where(peaks == 1)[0], marker='x', c='r') 
            # Draw and pause to update the plots  
            plt.draw()  
            plt.pause(0.01) 
            

    bag.close()