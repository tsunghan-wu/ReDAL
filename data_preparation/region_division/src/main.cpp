#include <iostream>
#include <unordered_map>
#include <boost/program_options.hpp>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "dataset.h"

using namespace std;
namespace po = boost::program_options;

typedef struct vccs_param{
    // default initial value
    float voxel_resolution = 0.15f;
    float seed_resolution = 3.5f;
    float color_importance = 0.0f;
    float spatial_importance = 1.0f;
    float normal_importance = 0.0f;

}VCCS_param;


void process_cloud(CloudXYZRGBA &cloud, VCCS_param &param, const fs::path &src, void (*write_supvox_func)(pcl::PointCloud<pcl::PointXYZL> &, const fs::path&)){
    //////////////////////////////  //////////////////////////////
    ////// This is how to use supervoxels
    //////////////////////////////  //////////////////////////////
    bool save_vis = false;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
    cloud_ptr = cloud.makeShared();

    pcl::SupervoxelClustering<pcl::PointXYZRGBA> super (param.voxel_resolution, param.seed_resolution);
    super.setInputCloud (cloud_ptr);
    super.setColorImportance (param.color_importance);
    super.setSpatialImportance (param.spatial_importance);
    super.setNormalImportance (param.normal_importance);

    map <std::uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr > supervoxel_clusters;

    super.extract (supervoxel_clusters);
    pcl::console::print_highlight ("Found %d supervoxels\n", supervoxel_clusters.size ());
  
    // save screenshot
    if (save_vis == true){
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor (0, 0, 0);

        CloudXYZRGBA::Ptr voxel_centroid_cloud = super.getVoxelCentroidCloud ();
        viewer->addPointCloud (voxel_centroid_cloud, "voxel centroids");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "voxel centroids");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.95, "voxel centroids");

        pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud();
        viewer->addPointCloud (labeled_voxel_cloud, "labeled voxels");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");

        PointNCloudT::Ptr sv_normal_cloud = super.makeSupervoxelNormalCloud (supervoxel_clusters);
        // We have this disabled so graph is easy to see, uncomment to see supervoxel normals
        // viewer->addPointCloudNormals<PointNormal> (sv_normal_cloud,1,0.05f, "supervoxel_normals");

        pcl::console::print_highlight ("Getting supervoxel adjacency\n");
        std::multimap<std::uint32_t, std::uint32_t> supervoxel_adjacency;
        super.getSupervoxelAdjacency (supervoxel_adjacency);
        // To make a graph of the supervoxel adjacency, we need to iterate through the supervoxel adjacency multimap
        for (auto label_itr = supervoxel_adjacency.cbegin (); label_itr != supervoxel_adjacency.cend (); ){
            // First get the label
            std::uint32_t supervoxel_label = label_itr->first;
            // Now get the supervoxel corresponding to the label
            pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr supervoxel = supervoxel_clusters.at (supervoxel_label);

            //Now we need to iterate through the adjacent supervoxels and make a point cloud of them
            CloudXYZRGBA adjacent_supervoxel_centers;
            for (auto adjacent_itr = supervoxel_adjacency.equal_range (supervoxel_label).first; adjacent_itr!=supervoxel_adjacency.equal_range (supervoxel_label).second; ++adjacent_itr)
            {
                pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr neighbor_supervoxel = supervoxel_clusters.at (adjacent_itr->second);
                adjacent_supervoxel_centers.push_back (neighbor_supervoxel->centroid_);
            }
            // Now we make a name for this polygon
            std::stringstream ss;
            ss << "supervoxel_" << supervoxel_label;
            // Move iterator forward to next label
            label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);
        }
        viewer->setCameraPosition(0, 0, 70, 0, 1, 0, 0);
        write_semkitti_visualize(viewer, src);
        viewer->close();
    }
    // save super voxel
    pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_cloud = super.getLabeledCloud();
    pcl::PointCloud<pcl::PointXYZL> output_cloud = *labeled_cloud;
    unordered_map<int, int> id_freq;
    for (int i = 0; i < output_cloud.size(); i++){
        unordered_map<int, int>::iterator it = id_freq.find(output_cloud[i].label);
        if (it != id_freq.end()){
            it->second += 1;
        }
        else{
            id_freq.insert(make_pair(output_cloud[i].label, 1));
        }
    }
    // prune < 10 points
    for (int i = 0; i < output_cloud.size(); i++){
        unordered_map<int, int>::iterator it = id_freq.find(output_cloud[i].label);
        if (it->second < 100){
            output_cloud[i].label = 0;
        }
    }
    // remap ids
    int new_id = 1;
    unordered_map<int, int> id_mapping;
    id_mapping.insert(make_pair(0, 0));
    for (int i = 0; i < output_cloud.size(); i++){
        unordered_map<int, int>::iterator it = id_mapping.find(output_cloud[i].label);
        if (it != id_mapping.end()){
            output_cloud[i].label = it->second;
        }
        else{
            id_mapping.insert(make_pair(output_cloud[i].label, new_id));
            output_cloud[i].label = new_id;
            new_id += 1;
        }
    }
    cout << new_id << endl;
    write_supvox_func(output_cloud, src);
    return;
}

int main(int argc, char *argv[]){
    // Get Argument
    bool save_vis = false;

    VCCS_param param = VCCS_param();

    string dataset, input_path, output_path;
    po::options_description desc("SuperVoxel Helper");
    desc.add_options()
        ("help,h", "produce help message")
        ("dataset", po::value<string>(&dataset), "Dataset Name, semantickitti / s3dis / scannet")
        ("input-path", po::value<string>(&input_path), "Input Directory")
        ("output-path", po::value<string>(&output_path), "Output Directory")
        ("voxel-resolution", po::value<float>(&(param.voxel_resolution)), "voxel resolution in VCCS algo.")
        ("seed-resolution", po::value<float>(&(param.seed_resolution)), "seed resolution in VCCS algo.")
        ("color-weight", po::value<float>(&(param.color_importance)), "color importance weight in VCCS algo.")
        ("spatial-weight", po::value<float>(&(param.spatial_importance)), "spatial importance weight in VCCS algo.")
        ("normal-weight", po::value<float>(&(param.normal_importance)), "normal importance weight in VCCS algo.")
        ("save-vis", po::bool_switch(&save_vis), "save visualization or not. Default : binary. e.g. --save-vis")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }
    // read dataset
    cout << input_path << endl;
    vector<string> input_fnames;
    if (dataset == "semantickitti")
        load_semkitti_list(input_path, input_fnames);
    else if (dataset == "s3dis")
        load_s3dis_list(input_path, input_fnames);
    else if (dataset == "scannet")
        load_scannet_list(input_path, input_fnames);
    // generate VCCS for each point cloud
    for (auto item : input_fnames){
        // load point cloud
        CloudXYZRGBA cloud;
        if (dataset == "semantickitti"){
            read_semkitti_velodyne(fs::path(item), cloud);
            process_cloud(cloud, param, fs::path(item), write_semkitti_supvoxel);
            //process_cloud(cloud, param, fs::path(item), save_vis);
        }
        else if (dataset == "s3dis"){
            read_s3dis_pc(fs::path(item), cloud);
            process_cloud(cloud, param, fs::path(item), write_s3dis_supvoxel);
        }
        else if (dataset == "scannet"){
            read_scannet_pc(fs::path(item), cloud);
            process_cloud(cloud, param, fs::path(item), write_scannet_supvoxel);
        }
    }
    return 0;
}
