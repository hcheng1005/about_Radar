from absl import app, flags, logging

import rosbag
import cv_bridge
import numpy as np
import cv2

import json

import tempfile
from pathlib import Path
import os
import shutil

from dataloader.radar import read_radar_params, rosmsg_to_radarcube_v1
from dataloader.radar import RadarFrameV1
from dataloader.h5_dataset import H5DatasetSaver, H5UnalignedDatasetSaver
from dataloader.stream_alignment import SensorStreamAlignment

from tqdm import tqdm

TOPICS_MAP = {'radar':'/radar_data',
              'depth':'/camera/aligned_depth_to_color/image_raw',
              'rgb':'/camera/color/image_raw',
              'imu_accel':'/camera/accel/sample',
              'imu_gyro':'/camera/gyro/sample',
             }

NAMES_MAP = { v:k for (k, v) in TOPICS_MAP.items() }

FLAGS = flags.FLAGS
flags.DEFINE_string('bagfile', None, 'rosbag file to open')
flags.DEFINE_list('topics', ['all'],
                  f' "all" or a comma separated list of streams to export [{",".join(TOPICS_MAP.keys())}]')
flags.DEFINE_string('radarcfg', None, 'Radar config file')
flags.DEFINE_string('h5file', None, 'Filename to save beamformed and temporally aligned frames to h5 file')
flags.DEFINE_bool('copy_to_tmp', False, "Copy to tmp folder, faster for bags on NFS")
flags.DEFINE_bool('export_unaligned', False,
                  'Do no perform stream alignment, export all samples with associated timestamps')

flags.mark_flag_as_required('bagfile')
flags.mark_flag_as_required('h5file')
flags.mark_flag_as_required('radarcfg')


def print_timestamps(frame_group):
    print('Grouped frames with timestamps:')
    t_ref = frame_group[0][1]
    #for f in frame_group:
    #    print('    ', f[1])

    print(f'  time delta: {(frame_group[1][1]-t_ref)/1e9}s (rgb) ')
    print(f'  time delta: {(frame_group[2][1]-t_ref)/1e9}s (depth) ')


def main(argv):
    logging.info("Starting export")
    if len(argv) > 1:
        print('Unprocessed args:', argv)

    logging.info("Loading rosbag")
    if FLAGS.copy_to_tmp:
        tempdir = tempfile.TemporaryDirectory()
        temp_bagfile = Path(tempdir.name)/(Path(FLAGS.bagfile).name)

        logging.info(f"Copying {FLAGS.bagfile} to {temp_bagfile}")
        shutil.copy(FLAGS.bagfile, temp_bagfile)
        logging.info('Copying done')
        bag = rosbag.Bag(temp_bagfile)
    else:
        bag = rosbag.Bag(FLAGS.bagfile)

    bridge = cv_bridge.CvBridge()

    #load radar config
    radar_cfg = read_radar_params(FLAGS.radarcfg)
    rf = RadarFrameV1(radar_cfg)
    logging.info(f"Radar max_range: {rf.max_range}m, resolution:{rf.range_resolution}m")

    if FLAGS.topics[0] == 'all':
        all_topics = list(bag.get_type_and_topic_info().topics.keys())
        stream_names = list([
            NAMES_MAP[t] for t in all_topics if t in NAMES_MAP.keys()
        ])
        #Move radar to first
        if 'radar' in stream_names:
            stream_names.insert(0, stream_names.pop(stream_names.index('radar')))
    else:
        stream_names = FLAGS.topics


    if FLAGS.export_unaligned:
        logging.info(f"Exporting unaligned {stream_names}")
        saver = H5UnalignedDatasetSaver(FLAGS.h5file, stream_names, include_timestamps=True)
        temporalAlignment = saver
    else:
        logging.info(f"Will be aligning {stream_names}")
        saver = H5DatasetSaver(FLAGS.h5file, stream_names, include_timestamps=True)
        remove_timestamp_add = lambda data: saver.push([d for d in data])
        temporalAlignment = SensorStreamAlignment(remove_timestamp_add, len(stream_names))

    topic_names = list([TOPICS_MAP[t] for t in stream_names])
    topic_to_stream_idx = { t: idx for (t, idx) in zip(topic_names, range(len(topic_names))) }

    coun = 0
    logging.info("Iterating through bag messages")
    for topic, msg, t in tqdm(bag.read_messages(topics=topic_names),
                              total=bag.get_message_count(topic_names)
                             ):
        if topic == '/camera/color/image_raw':
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            temporalAlignment.push_frame(img, t.to_nsec(), topic_to_stream_idx[topic])
        elif topic == '/camera/aligned_depth_to_color/image_raw':
            depth_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            temporalAlignment.push_frame(depth_img, t.to_nsec(), topic_to_stream_idx[topic])
        elif topic == '/radar_data' and FLAGS.radarcfg:
            radar_cube_raw = rosmsg_to_radarcube_v1(msg, radar_cfg)
            temporalAlignment.push_frame(radar_cube_raw, t.to_nsec(), topic_to_stream_idx[topic])
            coun +=1
        elif topic == '/camera/accel/sample':
            accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            temporalAlignment.push_frame(accel, t.to_nsec(), topic_to_stream_idx[topic])
        elif topic == '/camera/gyro/sample':
            gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            temporalAlignment.push_frame(gyro, t.to_nsec(), topic_to_stream_idx[topic])

        if coun > 10:
            break

    with open(f'{FLAGS.h5file}.json', "w") as f:
        json.dump(radar_cfg, f)



if __name__ == '__main__':
    # app.run(main)
    radar_cfg = read_radar_params('/home/charles/Code/RaDICal/demo/radarcfg/outdoor_human_rcs_50m.cfg')
    print(radar_cfg)
