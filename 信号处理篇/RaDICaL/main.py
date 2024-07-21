

import numpy as np  
import matplotlib.pyplot as plt  
from glob import glob

import rosbag  

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

    # Set up the plot  
    plt.rcParams['figure.figsize'] = [10, 10]  
    plt.ion()  
    fig, ax1 = plt.subplots()  

    # 遍历消息  
    for topic, msg, t in bag.read_messages(topics=['/radar_data']):  
        try:       
            # 提取雷达数据  
            radar_data = rosmsg_to_radarcube_v1(msg, radar_cfg)  
            radar_cube = dsp.range_processing(radar_data)
            fft2d_log_abs, aoa_input = dsp.doppler_processing(radar_cube,interleaved=False)
            
            # 可视化range-Doppler图像  
            plot_range_doppler(plt, ax1, fft2d_log_abs, f'Range-Doppler Map (T={t})')   
                
            # 保存为 NumPy 文件  
            if SAVE_FLAG:
                np.save(f'radar_data_{save_counter}.npy', radar_data)  


        except AttributeError as e:  
            print(f"Failed to read message: {e}")  

    bag.close()