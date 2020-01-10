#include <iostream>
#include <string>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc

#include <ctime>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/thread/thread.hpp>
#include <pcl/features/fpfh_omp.h> //包含fpfh加速计算的omp(多核并行计算)
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_features.h> 
#include <pcl/registration/correspondence_rejection_sample_consensus.h> 
#include <pcl/filters/approximate_voxel_grid.h>

typedef pcl::PointXYZ PointT;
typedef pcl::Normal NormalT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<NormalT> PointnormalT;
typedef pcl::PointCloud<pcl::FPFHSignature33> fpfhFeature;

void print4x4Matrix(const Eigen::Matrix4d & matrix)
{
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

fpfhFeature::Ptr compute_fpfh_feature(PointCloudT::Ptr input_cloud, pcl::search::KdTree<PointT>::Ptr tree)
{
	//normals
	PointnormalT::Ptr pointnormal(new PointnormalT);
	pcl::NormalEstimation<PointT, NormalT> est_normal;
	est_normal.setInputCloud(input_cloud);
	est_normal.setSearchMethod(tree);
	est_normal.setKSearch(10);
	est_normal.compute(*pointnormal);
	//fpfh 
	fpfhFeature::Ptr fpfh(new fpfhFeature);
	pcl::FPFHEstimationOMP<PointT, NormalT, pcl::FPFHSignature33> est_fpfh;
	est_fpfh.setNumberOfThreads(12); //指定12核计算
	est_fpfh.setInputCloud(input_cloud);
	est_fpfh.setInputNormals(pointnormal);
	est_fpfh.setSearchMethod(tree);
	est_fpfh.setKSearch(10);
	est_fpfh.compute(*fpfh);

	return fpfh;
}

int main()
{
	double spending_time = 0.0;
	PointCloudT::Ptr cloud_in(new PointCloudT);  // Original point cloud
	PointCloudT::Ptr cloud_in_copy(new PointCloudT);
	PointCloudT::Ptr cloud_tr(new PointCloudT);  // Rough transformed point cloud
	PointCloudT::Ptr cloud_icp(new PointCloudT);  // ICP output point cloud
	PointCloudT::Ptr cloud_icp_copy(new PointCloudT);
	PointCloudT::Ptr cloud_icpdone(new PointCloudT);

	char* P = "E:\\pg_one\\CG\\final_project\\ICP\\P0.obj";
	char* Q = "E:\\pg_one\\CG\\final_project\\ICP\\newQ0\\Q0.obj";
	int iterations = 15; 

	pcl::console::TicToc time;
	time.tic();
	pcl::io::loadOBJFile(Q, *cloud_in);
	pcl::io::loadOBJFile(P, *cloud_icp);
	std::cout << "Loading files takes " << (time.toc()/1000.0) << " s.\n" << std::endl;
	spending_time += (time.toc() / 1000.0);


    //downsample
	pcl::VoxelGrid<PointT> voxelSampler;
	voxelSampler.setInputCloud(cloud_in);
	voxelSampler.setLeafSize(3.0f, 3.0f, 3.0f);
	voxelSampler.filter(*cloud_in_copy);

	pcl::VoxelGrid<PointT> voxelSampler2;
	voxelSampler2.setInputCloud(cloud_icp);
	voxelSampler2.setLeafSize(3.0f, 3.0f, 3.0f);
	voxelSampler2.filter(*cloud_icp_copy);

	//特征粗匹配
	time.tic();
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	fpfhFeature::Ptr cloud_in_fpfh = compute_fpfh_feature(cloud_in_copy, tree);
	fpfhFeature::Ptr cloud_icp_fpfh = compute_fpfh_feature(cloud_icp_copy, tree);

	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
	sac_ia.setInputSource(cloud_icp_copy);
	sac_ia.setSourceFeatures(cloud_icp_fpfh);
	sac_ia.setInputTarget(cloud_in_copy);
	sac_ia.setTargetFeatures(cloud_in_fpfh);
	sac_ia.align(*cloud_icp_copy);

	std::cout << "Realizing rough regeistering takes " << (time.toc()/1000.0) << " s." << std::endl;
	spending_time += (time.toc() / 1000.0);
	Eigen::Matrix4d rough_transformation_matrix = sac_ia.getFinalTransformation().cast<double>();
	std::cout << "FPFH transformation: " << std::endl;
	print4x4Matrix(rough_transformation_matrix);
	pcl::transformPointCloud(*cloud_icp, *cloud_tr, rough_transformation_matrix);

    // The Iterative Closest Point algorithm
	time.tic();
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setMaximumIterations(iterations);
	icp.setInputSource(cloud_tr);
	icp.setInputTarget(cloud_in);
	icp.align(*cloud_icpdone);
	std::cout << "\nApplied " << iterations << " ICP iterations, it takes " << (time.toc()/1000.0) << " s." << std::endl;
	spending_time += (time.toc() / 1000.0);
	Eigen::Matrix4d icp_transformation_matrix;
	Eigen::Matrix4d transformation_matrix;
	if (icp.hasConverged())
	{
		std::cout << "ICP transformation: " << std::endl;
		icp_transformation_matrix = icp.getFinalTransformation().cast<double>();
		print4x4Matrix(icp_transformation_matrix);
		transformation_matrix = rough_transformation_matrix * icp_transformation_matrix;
		std::cout << "\nTotal transformation: " << std::endl;
		print4x4Matrix(transformation_matrix);
		std::cout << "The score is " << icp.getFitnessScore() << std::endl;
		std::cout << "The work totally takes " << spending_time << " s." << std::endl;
	}
	else
	{
		PCL_ERROR("\nICP has not converged.\n");
		return (-1);
	}

	// Visualization
	pcl::visualization::PCLVisualizer viewer("ICP demo");
	// Create two vertically separated viewports
	int v1(0);
	int v2(1);
	int v3(2);
	viewer.createViewPort(0.0, 0.0, 0.333, 1.0, v1);
	viewer.createViewPort(0.333, 0.0, 0.666, 1.0, v2);
	viewer.createViewPort(0.666, 0.0, 1.0, 1.0, v3);

	// The color we will be using
	float bckgr_gray_level = 0.0;  // Black
	float txt_gray_lvl = 1.0 - bckgr_gray_level;

	// Original point cloud is white
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h(cloud_in, (int)255 * txt_gray_lvl, (int)255 * txt_gray_lvl,
		(int)255 * txt_gray_lvl);
	viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v1", v1);
	viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v2", v2);
	viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v3", v3);

	// Original point cloud is blue
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_original_color_h(cloud_icp, 20, 20, 180);
	viewer.addPointCloud(cloud_icp, cloud_original_color_h, "cloud_original_v1", v1);

	// Transformed point cloud is green
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h(cloud_tr, 20, 180, 20);
	viewer.addPointCloud(cloud_tr, cloud_tr_color_h, "cloud_tr_v2", v2);

	// ICP aligned point cloud is red
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h(cloud_icpdone, 180, 20, 20);
	viewer.addPointCloud(cloud_icpdone, cloud_icp_color_h, "cloud_icp_v3", v3);

	// Set background color
	viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
	viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

	// Set camera position and orientation
	viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
	viewer.setSize(1280, 1024);  // Visualiser window size

	// Display the visualiser
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
	return (0);
}