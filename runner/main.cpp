#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cxxopts.hpp>
#include "FeedforwardNeuralNetwork.h"
#include "common/util.h"
#include "common/TrainSettings.h"
#include "common/TrainResult.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;

int main(int argc, char* argv[])
{
    cxxopts::Options options("runner",
      "A PoC implementation of a parallel Feed-forward neural network for CUDA");

    options.add_options()
      ("t,threads", "Number of threads to use", cxxopts::value<int>()->default_value("1"))
      ("b,blocks", "Number of blocks to use", cxxopts::value<int>()->default_value("1"))
      ("i,iterations", "Max iterations", cxxopts::value<int>()->default_value("1"))
      ("s,steps", "Inner steps (steps per thread)", cxxopts::value<int>()->default_value("1"))
      ("r,regularization_term", "Regularization term", cxxopts::value<double>()->default_value("0.0"))
      ("m,momentum", "Momentum", cxxopts::value<double>()->default_value("0.0"))
      ("l,learning_rate", "Learning rate", cxxopts::value<double>()->default_value("0.1"))
      ("e,epsilon", "During training, weights will be initialized between [-e, e]", cxxopts::value<double>()->default_value("10"))
      ("error", "Target error", cxxopts::value<double>()->default_value("0.00001"))
      ("h,help", "Print help")
      ;

    options.parse(argc, argv);

    if (options.count("help")) {
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }


    srand((unsigned int) time(0));

    std::random_device rd;
    std::mt19937 generator(rd());

    FeedforwardNeuralNetwork network({2, 8, 1});
    //FeedforwardNeuralNetwork network({3, 8, 1});

    /*MatrixXd weights(2, 3);
    weights << -30, 20, 20, 10, -20, -20;
    MatrixXd weights2(1, 3);
    weights2 << -10, 20, 20;
    network.set_weights(0, weights);
    network.set_weights(1, weights2);*/

    MatrixXd sample_x(4, 2);
    sample_x << 0, 0, 0, 1, 1, 0, 1, 1;

    MatrixXd sample_y(4, 1);
    sample_y << 0, 1, 1, 0;

    /*auto data = read_iris_database("iris.data");
    MatrixXd& sample_x = data.first;
    MatrixXd& sample_y = data.second;*/

    MatrixXd test_sample_x = sample_x;
    MatrixXd test_sample_y = sample_y;

    /*MatrixXd training_data = generate_data(1000, generator);
    MatrixXd test_data = generate_data(100, generator);
    MatrixXd sample_x = training_data.leftCols(2);
    MatrixXd sample_y = training_data.rightCols(1);
    MatrixXd test_sample_x = test_data.leftCols(2);
    MatrixXd test_sample_y = test_data.rightCols(1);*/

    TrainSettings train_settings;
    train_settings.threads = options["threads"].as<int>();
    train_settings.generator = &generator;
    train_settings.inner_steps = options["steps"].as<int>();
    train_settings.iterations = options["iterations"].as<int>();
    train_settings.initialize_weights_randomly = true;
    train_settings.regularization_term = options["regularization_term"].as<double>();
    train_settings.momentum = options["momentum"].as<double>();
    train_settings.step_factor = options["learning_rate"].as<double>();
    train_settings.random_epsilon = options["epsilon"].as<double>();
    train_settings.target_error = options["error"].as<double>();

    /*train_settings.regularization_term = 0.0;
    train_settings.momentum = 0.9;
    train_settings.step_factor = 1.0;
    train_settings.random_epsilon = 10;
    train_settings.target_error = 0.001;*/
    /*train_settings.regularization_term = 0.1;
  	train_settings.momentum = 0.6;
  	train_settings.step_factor = 0.06;
  	train_settings.random_epsilon = 10;
  	train_settings.target_error = 0.001;*/

    auto train_result = network.train(sample_x, sample_y, train_settings);

    for(int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            VectorXd input(2);
            input << i, j;
            auto result = network.predict(input);
            cout << "x: [" << i << ", " << j << "] => " << result[0] << '\n';
        }
    }

    cout << "Actual iterations: " << train_result.iterations << "\n";
    cout << "Training error: " << network.compute_error(sample_x,
                                                        sample_y,
                                                        train_settings.regularization_term) << "\n";
    cout << "Test error: " << network.compute_error(test_sample_x,
                                                    test_sample_y,
                                                    train_settings.regularization_term) << "\n";
}
