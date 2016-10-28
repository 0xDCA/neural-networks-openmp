#ifndef NEURAL_NETWORKS_FEEDFORWARDNEURALNETWORK_H
#define NEURAL_NETWORKS_FEEDFORWARDNEURALNETWORK_H

#include <vector>
#include <Eigen/Dense>
#include <stdexcept>
#include <exception>

#include "common/TrainResult.h"

struct TrainSettings;
struct WorkerParams {
    const Eigen::MatrixXd* x;
    const Eigen::MatrixXd* y;
    const TrainSettings* train_settings;
    std::vector<Eigen::MatrixXd>* weights;
    unsigned int seed;
};

class FeedforwardNeuralNetwork {
public:
    FeedforwardNeuralNetwork(const std::vector<int>& layers);

    Eigen::VectorXd predict(const Eigen::VectorXd& input) const;

    void set_weights(int source_layer, const Eigen::MatrixXd& weights);
    Eigen::MatrixXd get_weights(int source_layer) const;

    TrainResult train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, const TrainSettings& train_settings);

    double compute_error(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double regularization_term) const;

    std::vector<Eigen::MatrixXd> compute_weights_error(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y) const;

private:
    std::vector<int> layers;
    std::vector<Eigen::MatrixXd> weight_list;

    std::vector<Eigen::VectorXd> forward_propagation(const Eigen::VectorXd& input) const;

    Eigen::MatrixXd random_matrix(int rows, int cols, double epsilon, std::mt19937& generator);
    void back_propagation(const std::vector<Eigen::VectorXd>& fp_results, const Eigen::VectorXd& y,
                          std::vector<Eigen::MatrixXd>& out) const;

    static std::vector<Eigen::VectorXd> forward_propagation(const Eigen::VectorXd& input,
                                                            const std::vector<Eigen::MatrixXd>& weight_list);
    static void back_propagation(const std::vector<Eigen::VectorXd>& fp_results, const Eigen::VectorXd& y,
                          const std::vector<Eigen::MatrixXd>& weight_list,
                          std::vector<Eigen::MatrixXd>& out);

    static void do_gradient_descent(const WorkerParams& params);
    static double compute_error(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double regularization_term,
      const std::vector<Eigen::MatrixXd>& weight_list);
};


#endif //NEURAL_NETWORKS_FEEDFORWARDNEURALNETWORK_H
