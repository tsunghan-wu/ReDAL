#include "dataset.h"
#include <algorithm>

void load_semkitti_list(const string& input_dir, vector<string>& input_fnames){
    /*
        Parse SemanticKITTI dataset directory.
        Given "input_dir" and store in "input_fnames"

        File Organization
        input_dir (SemanticKitti/sequences)
        ├── 00
        │   ├── velodyne
        │   └── labels
        ├── 01
        ...
        
    */

    fs::path root_dir(input_dir);
    if (!fs::is_directory(root_dir)){
        std::cerr << root_dir.c_str() << "is not a directory" << std::endl;
    }
    // clear vector to play safe
    input_fnames.clear();
    std::vector<string> train_scans{"00", "01", "02", "03", "04", "05", "06", "07", "09", "10"};

    for(auto scan : train_scans){
        fs::path scan_root_dir = root_dir / scan / "velodyne";
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

void read_semkitti_velodyne(const fs::path& fileName, CloudXYZRGBA& cloud){
    std::ifstream input(fileName.c_str(), std::ios_base::binary);
    if(!input.good()){
        std::cerr<<"Cannot open file : "<<fileName<<std::endl;
        return;
    }


    cloud.clear();
    cloud.height = 1;
    float tmp;
    pcl::PointXYZRGBA point;
    point.r = 0, point.g = 0, point.b = 0, point.a = 0;
    while(input.read((char *) &point.x, 3*sizeof(float))){
        input.read((char *) &tmp, sizeof(float));
        // std::cerr << "file pointer: " << input.tellg() << "bytes" << std::endl;
        cloud.push_back(point);
    }
    std::cerr << fileName.stem() << ":" << cloud.width << " points" << std::endl;
    input.close();
    return;
}

void write_semkitti_supvoxel(pcl::PointCloud<pcl::PointXYZL>& output_cloud, const fs::path& srcpath){
    fs::path fname = srcpath.filename();
    fs::path scan_dir = srcpath.parent_path().parent_path();
    fs::path new_dir = scan_dir / "supervoxel";
    fs::create_directory(new_dir);  // create directory if not exists
    fs::path fileName = new_dir / fname;

    // write point cloud to binary
    std::ofstream output(fileName.c_str(), std::ios_base::binary);
    if (!output.good()){
        std::cerr<<"Cannot open file : " << fileName << std::endl;
    }
    // pcl::PointCloud<pcl::PointXYZL> output_cloud = *cloud;
    int zero_cnt = 0;
    for (int i =0; i < output_cloud.size(); i++){
        int l = (int) output_cloud[i].label;
        output.write(reinterpret_cast<const char*>(&l), sizeof(int));
        if(l == 0){
            zero_cnt += 1;
        }
    }
    cout << (float)zero_cnt / output_cloud.size() << endl;
    output.close();
    return;
}

void write_semkitti_visualize(pcl::visualization::PCLVisualizer::Ptr viewer, const fs::path& srcpath) {
    string fname = srcpath.stem().string() + ".png";
    fs::path scan_dir = srcpath.parent_path().parent_path();
    fs::path new_dir = scan_dir / "visualization";
    fs::create_directory(new_dir);  // create directory if not exists
    fs::path fileName = new_dir / fname;
    viewer->saveScreenshot(fileName.c_str());
    return;
}
