ROS to hdf5 converter and stream alignment
==========================================

Exports .bag files to timestamp aligned hdf5 files.


## How to use

Edit `exporth5.sh` to set variables so that they point to the correct paths were the dataset is
stored.

Run `exporth5.sh` to convert all bag files to a timestamp aligned hdf5 file. The correct radar
config will be picked up based on the path naming conventions. The resultant h5 file will be saved
in the same folder as the bag files, with `.export.h5` appended to the filename.

