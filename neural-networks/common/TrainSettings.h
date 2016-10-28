#ifndef NEURAL_NETWORKS_TRAINSETTINGS_H
#define NEURAL_NETWORKS_TRAINSETTINGS_H

struct TrainSettings {
    int iterations;
    double regularization_term;
    double step_factor;
    double momentum;
    int inner_steps;
    int threads;
    std::mt19937* generator;
    double random_epsilon;
    bool initialize_weights_randomly = true;
    double target_error;

    void validate() const {
        if (threads <= 0) {
            throw std::runtime_error("Invalid threads");
        }

        if (inner_steps <= 0) {
            throw std::runtime_error("Invalid inner_steps");
        }

        if (iterations <= 0) {
            throw std::runtime_error("Invalid iterations");
        }

        if (generator == nullptr) {
          throw std::runtime_error("Invalid generator");
        }

        if (target_error < 0) {
          throw std::runtime_error("Invalid target_error");
        }
    }
};

#endif //NEURAL_NETWORKS_TRAINSETTINGS_H
