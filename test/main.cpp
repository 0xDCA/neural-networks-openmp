#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include <Eigen/Dense>
#include "catch.hpp"
#include "FeedforwardNeuralNetwork.h"
#include "ApproxMatrixMatcher.h"
#include "common/TrainSettings.h"
#include "common/TrainResult.h"

using std::vector;

constexpr double sigmoid(double t) {
	return 1.0 / (1 + std::exp(-t));
}

template <class TElem>
std::string to_string(const vector<TElem>& x) {
	std::ostringstream os;

	os << "[";
	for(size_t i = 0; i < x.size(); ++i) {
		if (i >= 1) {
			os << ' ';
		}

		os << x[i];
	}

	os << ']';

	return os.str();
}

Eigen::VectorXd to_vectorXd(const vector<double>& source) {
	Eigen::VectorXd result(source.size());

	for (size_t i = 0; i < source.size(); ++i) {
		result[i] = source[i];
	}

	return result;
}

TEST_CASE("can set up weights manually") {
	FeedforwardNeuralNetwork network({2, 4, 1});

	Eigen::MatrixXd weights(4, 3);

	for(int i = 0; i < weights.rows(); ++i) {
		for (int j = 0; j < weights.cols(); ++j) {
			weights(i, j) = 1.0;
		}
	}

	network.set_weights(0, weights);

	REQUIRE(network.get_weights(0) == weights);
}

void test_single_output_neural_network(const FeedforwardNeuralNetwork& network,
		const vector<vector<double> >& inputs, const vector<double>& expected_outputs) {
	REQUIRE(inputs.size() == expected_outputs.size());

	for(size_t i = 0; i < inputs.size(); ++i) {
		CAPTURE(i);
		CAPTURE(to_string(inputs[i]));

		auto input = to_vectorXd(inputs[i]);
		auto output = network.predict(input);

		REQUIRE(output.size() == 1);
		REQUIRE(output[0] == Approx(expected_outputs[i]));
	}
}

void test_single_output_neural_network_approx(const FeedforwardNeuralNetwork& network,
		const vector<vector<double> >& inputs, const vector<bool>& expected_outputs, double threshold = 0.8) {
	REQUIRE(inputs.size() == expected_outputs.size());

	for(size_t i = 0; i < inputs.size(); ++i) {
		CAPTURE(i);
		CAPTURE(to_string(inputs[i]));

		auto input = to_vectorXd(inputs[i]);
		auto output = network.predict(input);

		REQUIRE(output.size() == 1);
		if (expected_outputs[i]) {
			REQUIRE(output[0] >= threshold);
		} else {
			REQUIRE(output[0] < threshold);
		}

	}
}

TEST_CASE("performs forward-propagation correctly", "[forwardpropagation]") {
	SECTION("AND gate neural-network") {
		FeedforwardNeuralNetwork network({2, 1});
		Eigen::MatrixXd weights(1, 3);
		weights <<
			-30, 20, 20;

		network.set_weights(0, weights);

		test_single_output_neural_network(network, { {0, 0}, {0, 1}, {1, 0}, {1, 1} },
				{sigmoid(-30), sigmoid(-10), sigmoid(-10), sigmoid(10)});
	}

	SECTION("OR gate neural-network") {
		FeedforwardNeuralNetwork network({2, 1});
		Eigen::MatrixXd weights(1, 3);
		weights <<
			-10, 20, 20
		;

		network.set_weights(0, weights);

		test_single_output_neural_network(network, { {0, 0}, {0, 1}, {1, 0}, {1, 1} },
				{sigmoid(-10), sigmoid(10), sigmoid(10), sigmoid(30)});
	}

	SECTION("NOR gate neural-network") {
		FeedforwardNeuralNetwork network({2, 1});
		Eigen::MatrixXd weights(1, 3);
		weights <<
			10, -20, -20;

		network.set_weights(0, weights);

		test_single_output_neural_network(network, { {0, 0}, {0, 1}, {1, 0}, {1, 1} },
				{sigmoid(10), sigmoid(-10), sigmoid(-10), sigmoid(-30)});
	}

	SECTION("XOR gate neural-network") {
		FeedforwardNeuralNetwork network({2, 2, 1});

		Eigen::MatrixXd weights_0(2, 3);
		weights_0 <<
			30, -20, -20,
			-10, 20, 20
		;

		Eigen::MatrixXd weights_1(1, 3);
		weights_1 <<
			-30, 20, 20
		;

		network.set_weights(0, weights_0);
		network.set_weights(1, weights_1);

		test_single_output_neural_network_approx(network, { {0, 0}, {0, 1}, {1, 0}, {1, 1} },
				{false, true, true, false});
	}
}

TEST_CASE("performs back-propagation correctly", "[backpropagation]") {
	SECTION("simple neural network") {
		FeedforwardNeuralNetwork network({2, 2, 1});

		Eigen::MatrixXd weights_0(2, 3);
		weights_0 <<
			1, -5, -5,
			-5, 10, -10
		;

		Eigen::MatrixXd weights_1(1, 3);
		weights_1 <<
			-5, 1, 1
		;

		network.set_weights(0, weights_0);
		network.set_weights(1, weights_1);

		vector<Eigen::MatrixXd> expected_deltas;
		expected_deltas.push_back(Eigen::MatrixXd(2, 3));
		expected_deltas[0] <<
			-1.75423615e-02,  -0.00000000e+00,  -1.75423615e-02,
			-3.03817871e-07,  -0.00000000e+00,  -3.03817871e-07
		;

		expected_deltas.push_back(Eigen::MatrixXd(1, 3));
		expected_deltas[1] <<
			-9.93186507e-01,  -1.78636610e-02,  -3.03817964e-07
		;

		Eigen::VectorXd input(2);
		input << 0, 1;

		Eigen::VectorXd output(1);
		output << 1;

		vector<Eigen::MatrixXd> result = network.compute_weights_error(input, output);

		REQUIRE(result.size() == expected_deltas.size());
		for(size_t i = 0; i < result.size(); ++i) {
			CAPTURE(i);
			CHECK_THAT(result[i], ApproxMatrixMatcher(expected_deltas[i]));
		}
	}
}

TEST_CASE("calculates the error correctly", "[error]") {
	SECTION("simple neural network") {
		FeedforwardNeuralNetwork network({2, 2, 1});

		Eigen::MatrixXd sample_x(4, 2);
		sample_x <<
			0, 0,
			0, 1,
			1, 0,
			1, 1
		;

		Eigen::MatrixXd sample_y(4, 1);
		sample_y <<
			0,
			1,
			1,
			0
		;

		Eigen::MatrixXd weights_0(2, 3);
		weights_0 <<
			1, -5, -5,
			-5, 10, -10
		;

		Eigen::MatrixXd weights_1(1, 3);
		weights_1 <<
			-5, 1, 1
		;

		network.set_weights(0, weights_0);
		network.set_weights(1, weights_1);

		const double expected_error = 8.06265;

		SECTION("CPU error") {
			double actual_error = network.compute_error(sample_x, sample_y, 0.2);

			REQUIRE(actual_error == Approx(expected_error));
		}
	}
}

TEST_CASE("trains correctly", "[train]") {
	SECTION("XOR neural network") {
		FeedforwardNeuralNetwork network({2, 8, 1});

		Eigen::MatrixXd sample_x(4, 2);
		sample_x <<
			0, 0,
			0, 1,
			1, 0,
			1, 1;

		Eigen::MatrixXd sample_y(4, 1);
		sample_y <<
			0,
			1,
			1,
			0
		;

		std::random_device rd;
		std::mt19937 generator(rd());

		TrainSettings train_settings;
		train_settings.initialize_weights_randomly = true;
		train_settings.inner_steps = 16;
		train_settings.iterations = 2000;
		train_settings.regularization_term = 0.0;
		train_settings.momentum = 0.9;
		train_settings.step_factor = 1.0;
		train_settings.threads = 8;
		train_settings.generator = &generator;
		train_settings.random_epsilon = 10.0;
		train_settings.target_error = 0.001;

		TrainResult result = network.train(sample_x, sample_y, train_settings);

		CAPTURE(result.error);

		test_single_output_neural_network_approx(network, { {0, 0}, {0, 1}, {1, 0}, {1, 1} },
				{0, 1, 1, 0});
	}
}
