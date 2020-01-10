#include <iostream>
#include <string>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <ctime>
#include <Eigen/Core>
#include <Eigen/SVD>  
#include <Eigen/Dense> 
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
typedef pcl::PointCloud<PointT> PointCloudT;

int main()
{
	PointCloudT::Ptr cloud_in(new PointCloudT);  // Original point cloud
	PointCloudT::Ptr cloud_icp(new PointCloudT);  // ICP output point cloud
	PointCloudT::Ptr cloud_tr(new PointCloudT);

	char* P = "E:\\pg_one\\CG\\final_project\\ICP\\P0.obj";
	char* Q = "E:\\pg_one\\CG\\final_project\\ICP\\newQ0\\Q0.obj";

	clock_t startTime, endTime;
	startTime = clock();
	pcl::io::loadOBJFile(Q, *cloud_in);
	pcl::io::loadOBJFile(P, *cloud_icp);
	*cloud_tr = *cloud_icp;
	endTime = clock();
	std::cout << "Loading files takes " << (double)(endTime - startTime) / CLOCKS_PER_SEC << " s." << std::endl;

	const int POINT_NUM = cloud_in->points.size();
	const int SELECT_NUM = 10;
	const int ITER_NUM = 200;

	Eigen::MatrixXf T(1, 3);
	Eigen::MatrixXf R(3, 3);
	Eigen::MatrixXf P0(POINT_NUM, 3);
	Eigen::MatrixXf Q0(POINT_NUM, 3);

	for (int i = 0; i < POINT_NUM; i++) {
		P0(i, 0) = cloud_icp->points[i].x;
		P0(i, 1) = cloud_icp->points[i].y;
		P0(i, 2) = cloud_icp->points[i].z;
		Q0(i, 0) = cloud_in->points[i].x;
		Q0(i, 1) = cloud_in->points[i].y;
		Q0(i, 2) = cloud_in->points[i].z;
	}

	srand((unsigned int)(time(NULL)));
	startTime = clock();
	int k = 0;
	float error = 0;
	for (k = 0; k < ITER_NUM; k++) {
		Eigen::VectorXi mSelected(SELECT_NUM);
		for (int m = 0; m < SELECT_NUM; m++) {
			mSelected(m) = (int)(((double)rand() / RAND_MAX) * (POINT_NUM - 1));
		}
		Eigen::MatrixXf P0Select(SELECT_NUM, 3);
		for (int m = 0; m < SELECT_NUM; m++) {
			P0Select.row(m) = P0.row(mSelected(m));
		}

		int selectindex[SELECT_NUM] = { 0 };
		float distance_COMP;
		for (int i = 0; i < SELECT_NUM; i++) {
			distance_COMP = pow((P0Select(i, 0) - Q0(0, 0)), 2) + pow(P0Select(i, 1) - Q0(0, 1), 2) + pow(P0Select(i, 2) - Q0(0, 2), 2);
			for (int j = 0; j < POINT_NUM; j++) {
				float distance_square = pow((P0Select(i, 0) - Q0(j, 0)), 2) + pow(P0Select(i, 1) - Q0(j, 1), 2) + pow(P0Select(i, 2) - Q0(j, 2), 2);
				if (distance_square <= distance_COMP) {
					distance_COMP = distance_square;
					selectindex[i] = j;
				}
			}
		}
		Eigen::MatrixXf Q0Select(SELECT_NUM, 3);
		for (int i = 0; i < SELECT_NUM; i++) {
			Q0Select.row(i) = Q0.row(selectindex[i]);
		}
		Eigen::MatrixXf P0selectCentroid(1, 3);
		Eigen::MatrixXf Q0selectCentroid(1, 3);

		for (int i = 0; i < 3; i++) {
			P0selectCentroid(0, i) = P0Select.col(i).sum() / SELECT_NUM;
			Q0selectCentroid(0, i) = Q0Select.col(i).sum() / SELECT_NUM;
		}
		for (int i = 0; i < SELECT_NUM; i++) {
			P0Select.row(i) = P0Select.row(i) - P0selectCentroid;
			Q0Select.row(i) = Q0Select.row(i) - Q0selectCentroid;
		}
		Eigen::MatrixXf H = P0Select.transpose() * Q0Select;
		Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3f V = svd.matrixV(), U = svd.matrixU();
		R = U * V.transpose();
		P0 = P0 * R;
		T = Q0selectCentroid - P0selectCentroid * R;
		for (int i = 0; i < POINT_NUM; i++) {
			P0.row(i) = P0.row(i) + T;
		}
		if (((k + 5) % 20) == 0) {
			cout << "第" << k << "次迭代结束, ";
			for (int i = 0; i < POINT_NUM; i++) {
				cloud_icp->points[i].x = P0(i, 0);
				cloud_icp->points[i].y = P0(i, 1);
				cloud_icp->points[i].z = P0(i, 2);
			}
			error = 0;
			pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
			kdtree.setInputCloud(cloud_in);
			std::vector<int> pointIdxNKNSearch(1);
			std::vector<float> pointNKNSquaredDistance(1);
			for (int m = 0; m < POINT_NUM; m++) {
				kdtree.nearestKSearch(cloud_icp->points[m], 1, pointIdxNKNSearch, pointNKNSquaredDistance);
				error += pointNKNSquaredDistance[0];
			}
			error /= POINT_NUM;
			cout << "the error is: " << error << endl;
			if (error < 0.1)
				break;
		}
		else
			cout << "第" << k << "次迭代结束 " << endl;
	}
	endTime = clock();
	cout << "The running time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

	// Visualization
	pcl::visualization::PCLVisualizer viewer("ICP demo");
	// Create two vertically separated viewports
	int v1(0);
	int v2(1);

	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);


	// The color we will be using
	float bckgr_gray_level = 0.0;  // Black
	float txt_gray_lvl = 1.0 - bckgr_gray_level;

	// Original point cloud is white
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h(cloud_in, (int)255 * txt_gray_lvl, (int)255 * txt_gray_lvl,
		(int)255 * txt_gray_lvl);
	viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v1", v1);
	viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v2", v2);

	// Original point cloud is blue
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_original_color_h(cloud_tr, 20, 20, 180);
	viewer.addPointCloud(cloud_tr, cloud_original_color_h, "cloud_original_v1", v1);

	// Transformed point cloud is green
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h(cloud_icp, 20, 180, 20);
	viewer.addPointCloud(cloud_icp, cloud_icp_color_h, "cloud_tr_v2", v2);


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