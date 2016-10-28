#include <cmath>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <endian.h>

#include "data-util.h"

using Eigen::MatrixXd;
using namespace std;

Eigen::MatrixXd generate_data(int n, std::mt19937& generator) {
    Eigen::MatrixXd result(n, 3);
    std::uniform_real_distribution<> distribution(-1000.0, 1000.0);

    for (int i = 0; i < n; ++i) {
        double x1 = distribution(generator);
        double x2 = distribution(generator);

        double y = x2*x2 + x1 >= 5000 ? 1.0 : 0.0;

        result(i, 0) = x1;
        result(i, 1) = x2;
        result(i, 2) = y;
    }

    return result;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> read_mnist_database(const std::string& image_file_name, const std::string& label_file_name) {
  ifstream image_data (image_file_name, ios::in | ios::binary);
  ifstream label_data (label_file_name, ios::in | ios::binary);

  image_data.seekg(0x04);

  int examples = 0, rows = 0, cols = 0;

  image_data.read((char*)&examples, 4);
  image_data.read((char*)&rows, 4);
  image_data.read((char*)&cols, 4);

  examples = be32toh(examples);
  rows = be32toh(rows);
  cols = be32toh(cols);

  int dimensions = rows * cols;

  cout << "Reading " << examples << " examples (" << rows << " x " << cols << ")\n";

  MatrixXd x(examples, dimensions);
  for(int i = 0; i < examples; ++i) {
    for(int j = 0; j < dimensions; ++j) {
      unsigned char pixel;
      image_data.read((char*)&pixel, 1);

      x(i, j) = pixel;
    }
  }

  MatrixXd y(examples, 1);
  label_data.seekg(0x08);
  for(int i = 0; i < examples; ++i) {
    unsigned char label;
    label_data.read((char*)&label, 1);

    y(i, 0) = label;
  }

  cout << "Done reading\n";

  return make_pair(x, y);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> read_iris_database(const std::string& file_name) {
  const int examples = 150;
  const int input_columns = 3;
  const int output_columns = 1;

  cout << "Reading " << examples << " examples\n";

  ifstream data(file_name);

  MatrixXd x(examples, input_columns);
  MatrixXd y(examples, output_columns);

  for(int i = 0; i < examples; ++i) {
    for(int j = 0; j < input_columns; ++j) {
      double value;
      data >> value;

      x(i, j) = value;
    }

    for(int j = 0; j < output_columns; ++j) {
      double value;
      data >> value;

      y(i, j) = value;
    }
  }

  cout << "Done reading\n";

  return make_pair(x, y);
}
