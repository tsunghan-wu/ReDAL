#include "cnpy.h"
#include "dataset.h"
#include <algorithm>
#include <vector>
#include <regex>
#include <complex>

void load_s3dis_list(const string& input_dir, vector<string>& input_fnames){
    /*
        Parse S3DIS dataset directory.
        Given "input_dir" and store in "input_fnames"

        File Organization
        input_dir
        ├── Area_1
        │   ├── coords
        │   ├── labels
        │   └── rgb
        ├── Area_2
        ...
    */

    fs::path root_dir(input_dir);
    if (!fs::is_directory(root_dir)){
        std::cerr << root_dir.c_str() << "is not a directory" << std::endl;
    }
    // clear vector to play safe
    input_fnames.clear();
    std::vector<string> train_scans{"Area_1", "Area_2", "Area_3", "Area_4", "Area_6"};

    for(auto scan : train_scans){
        fs::path scan_root_dir = root_dir / scan / "coords";
        if (!fs::is_directory(scan_root_dir)){
            std::cerr << scan_root_dir.c_str() << "is not a directory" << std::endl;
        }
        // iterate all files in directory
        for(auto & p : fs::directory_iterator(scan_root_dir)){
            input_fnames.push_back(p.path().string());
        }
    }
    std::sort(input_fnames.begin(), input_fnames.end());
    std::cout << "Total " << input_fnames.size() << " files are found" << std::endl;
    return;
}

void read_s3dis_pc(const fs::path& fileName, CloudXYZRGBA& cloud){
    // filename
    std::string coords_fname = fileName.c_str();
    std::string rgb_fname = std::regex_replace(coords_fname, std::regex("coords"), "rgb");
    // load numpy array
    cnpy::NpyArray coords = cnpy::npy_load(coords_fname);
    cnpy::NpyArray rgb = cnpy::npy_load(rgb_fname);

    float* loaded_coords = coords.data<float>();
    uint8_t* loaded_rgb = rgb.data<uint8_t>();

    std::vector<size_t> shape = coords.shape;
    int N = coords.shape[0];

    // store into pointcloud
    cloud.clear();
    cloud.height = 1;
    pcl::PointXYZRGBA point;
    for(int i = 0; i < N; i++){
        point.x = loaded_coords[i*3];
        point.y = loaded_coords[i*3+1];
        point.z = loaded_coords[i*3+2];
        point.r = loaded_rgb[i*3];
        point.g = loaded_rgb[i*3+1];
        point.b = loaded_rgb[i*3+2];
        cloud.push_back(point);
    }
}

void write_s3dis_supvoxel(pcl::PointCloud<pcl::PointXYZL>& output_cloud, const fs::path& srcpath){
    fs::path fname = srcpath.filename();
    fs::path scan_dir = srcpath.parent_path().parent_path();
    fs::path new_dir = scan_dir / "supervoxel";
    fs::create_directory(new_dir);  // create directory if not exists
    fs::path fileName = new_dir / fname;

    // write point cloud to npy
    // pcl::PointCloud<pcl::PointXYZL> output_cloud = *cloud;
    std::vector<int> supvox;
    for (int i =0; i < output_cloud.size(); i++){
        int l = (int) output_cloud[i].label;
        supvox.push_back(l);
    }
    cnpy::npy_save(fileName.c_str(), &supvox[0], {output_cloud.size()}, "w");
    return;
}

