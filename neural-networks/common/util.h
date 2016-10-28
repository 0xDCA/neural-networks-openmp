
#ifndef NEURAL_NETWORKS_UTIL_H
#define NEURAL_NETWORKS_UTIL_H

#include <string>
#include <Eigen/Dense>

double sigmoid(double t);

std::string get_matrix_string(const Eigen::MatrixXd& matrix);

#endif //NEURAL_NETWORKS_UTIL_H
