#ifndef NEURAL_NETWORKS_TRAINRESULT_H
#define NEURAL_NETWORKS_TRAINRESULT_H

struct TrainResult {
    TrainResult(int iterations, double error) : iterations(iterations), error(error) {}

    int iterations;
    double error;
};

#endif //NEURAL_NETWORKS_TRAINRESULT_H
