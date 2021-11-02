#include "cnpy.h"
#include "dataset.h"
#include <algorithm>
#include <vector>
#include <regex>
#include <complex>

bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs)
{
    // Open the File
    std::ifstream in(fileName.c_str());
    // Check if object is valid
    if(!in)
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
        return false;
    }
    std::string str;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            vecOfStrs.push_back(str);
    }
    //Close The File
    in.close();
    return true;
}
void load_scannet_list(const string& input_dir, vector<string>& input_fnames){
    /*
        Parse scannetv2 dataset directory.
        Given "input_dir" and store in "input_fnames"
    */

    fs::path root_dir(input_dir);
    if (!fs::is_directory(root_dir)){
        std::cerr << root_dir.c_str() << "is not a directory" << std::endl;
    }
    // clear vector to play safe
    input_fnames.clear();
    std::vector<string> train_scans;
    bool result = getFileContent("scannetv2_train.txt",  train_scans);
    for(auto scan : train_scans){
        fs::path fname_path = root_dir / scan / "coords.npy";
        input_fnames.push_back(fname_path.string());
    }
    std::sort(input_fnames.begin(), input_fnames.end());
    std::cout << "Total " << input_fnames.size() << " files are found" << std::endl;
    return;
}

void read_scannet_pc(const fs::path& fileName, CloudXYZRGBA& cloud){
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

void write_scannet_supvoxel(pcl::PointCloud<pcl::PointXYZL>& output_cloud, const fs::path& srcpath){
    std::string fname = srcpath.c_str();
    // fs::path scan_dir = srcpath.parent_path().parent_path();
    // fs::path new_dir = scan_dir / "supervoxel";
    // fs::create_directory(new_dir);  // create directory if not exists
    // fs::path fileName = new_dir / fname;

    std::string fileName = std::regex_replace(fname, std::regex("coords"), "supervoxel");
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
