for folder in ../kittidata/*; do
    if [ ! -d $folder/velodyne_points/pcd_data ]; then
        mkdir $folder/velodyne_points/pcd_data
    fi
    for file in $folder/velodyne_points/data/*; do
        if [ ! -f $folder/velodyne_points/pcd_data/$(basename $file .bin).pcd ]; then
            ./kitti2pcd --infile $file --outfile $folder/velodyne_points/pcd_data/$(basename $file .bin).pcd
        fi
    done
done
