
#include <cmath>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <endian.h>

using Eigen::MatrixXd;
using namespace std;

double sigmoid(double t) {
    return 1.0 / (1 + exp(-t));
}

std::string get_matrix_string(const Eigen::MatrixXd &matrix) {
    std::ostringstream stream;
    stream << matrix;

    return stream.str();
}
