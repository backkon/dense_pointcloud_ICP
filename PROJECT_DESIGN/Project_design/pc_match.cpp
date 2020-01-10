#include <Eigen/Core>
#include <Eigen/SVD>  
#include <Eigen/Dense>   
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

const int POINT_NUM = 453089;
const int SELECT_NUM = 50;
const int ITER_NUM = 200;
using namespace std;


Eigen::MatrixXf LoadObjFile(const char* filename) {
	int i = 0;
	Eigen::MatrixXf pc(POINT_NUM, 3);
	std::ifstream ifs(filename);

	std::string line;
	while (std::getline(ifs, line)) {
		if (line.empty()) continue;

		std::istringstream iss(line);
		std::string type;
		iss >> type;
		// vertex
		if (type.compare("v") == 0) {
			iss >> pc(i,0) >> pc(i, 1) >> pc(i, 2);
			i++;
		}
		else if (type.compare("f") == 0) {
			break;
		}
	}
	ifs.close();
	cout << "Successful loading, " << "the size is " << i << "." << endl;
	return pc;
}

int main()
{
	Eigen::MatrixXf T(1, 3);
	Eigen::MatrixXf R(3, 3);
	Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f Trans_matrix = Eigen::Matrix4f::Identity();
	clock_t startTime, endTime;
	Eigen::MatrixXf P0 = LoadObjFile("E:\\pg_one\\CG\\final_project\\ICP\\P0.obj");
	Eigen::MatrixXf Q0 = LoadObjFile("E:\\pg_one\\CG\\final_project\\ICP\\Q0.obj");
	Eigen::MatrixXf Q0Norm(SELECT_NUM,POINT_NUM);
	for (int i = 0; i < POINT_NUM; i++){
		for (int j = 0; j < SELECT_NUM; j++) {
			Q0Norm(j, i) = Q0(i, 0)*Q0(i, 0) + Q0(i, 1)*Q0(i, 1) + Q0(i, 2)*Q0(i, 2);
		}
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
			//P0Select(m, 0) = P0(mSelected(m), 0);
			//P0Select(m, 1) = P0(mSelected(m), 1);
			//P0Select(m, 2) = P0(mSelected(m), 2);
			P0Select.row(m) = P0.row(mSelected(m));
		}
		Eigen::MatrixXf P0SelectNorm(SELECT_NUM, POINT_NUM);
		for (int i = 0; i < SELECT_NUM; i++) {
			for (int j = 0; j < POINT_NUM; j++) {
				P0SelectNorm(i, j) = P0Select(i, 0)*P0Select(i, 0) + P0Select(i, 1)*P0Select(i, 1) + P0Select(i, 2)*P0Select(i, 2);
			}
		}
		Eigen::MatrixXf dotMatrix = P0Select*(Q0.transpose());
		Eigen::MatrixXf disMatrix = P0SelectNorm + Q0Norm - 2 * dotMatrix;
		Eigen::Index selectindex[SELECT_NUM];
		for (int i = 0; i < SELECT_NUM; i++) {
			disMatrix.row(i).minCoeff(&selectindex[i]);
		}
		Eigen::MatrixXf Q0Select(SELECT_NUM, 3);
		for (int i = 0; i < SELECT_NUM; i++) {
			Q0Select.row(i) = Q0.row(selectindex[i]);
		}
		Eigen::MatrixXf P0selectCentroid(SELECT_NUM, 3);
		Eigen::MatrixXf Q0selectCentroid(SELECT_NUM, 3);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < SELECT_NUM; j++) {
				P0selectCentroid(j, i) = P0Select.col(i).sum() / SELECT_NUM;
				Q0selectCentroid(j, i) = Q0Select.col(i).sum() / SELECT_NUM;
			}
		}
		P0Select = P0Select - P0selectCentroid;
		Q0Select = Q0Select - Q0selectCentroid;
		Eigen::MatrixXf H = P0Select.transpose() * Q0Select;
		Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3f V = svd.matrixV(), U = svd.matrixU();
		R = U * V.transpose();
		P0 = P0 * R;
		T = Q0selectCentroid.row(0) - P0selectCentroid.row(0) * R;
		for (int i = 0; i < POINT_NUM; i++) {
			P0.row(i) = P0.row(i) + T;
		}
		error = 0;
		for (int i = 0; i < POINT_NUM; i++) {
			error += pow((Q0(i, 0) - P0(i, 0)), 2) + pow((Q0(i, 1) - P0(i, 1)), 2) + pow((Q0(i, 2) - P0(i, 2)), 2);
		}
		matrix.topLeftCorner(3, 3) << R;
		matrix.topRightCorner(3, 1) << T.transpose();
		Trans_matrix = Trans_matrix * matrix;
		cout << "第" << k << "次迭代结束:" << endl;
		cout << "The matrix is: " << endl << matrix << endl;
		error /= POINT_NUM;
		cout << "The error is: " << error << endl;
		if (error < 0.5)
			break;
	}
	endTime = clock();
	cout << endl << "Final results: " << endl;
	cout << "The running time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << "The transformation matrix is: " << endl << Trans_matrix << endl;
	cout << "The iter is: " << k << endl;
	cout << "The error is: " << error << endl;
	while (1);
	return 0;
}