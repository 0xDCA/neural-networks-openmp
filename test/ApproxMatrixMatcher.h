#ifndef APPROXMATRIXMATCHER_H_
#define APPROXMATRIXMATCHER_H_

#include <Eigen/Dense>
#include <catch.hpp>
#include <string>
#include <limits>

class ApproxMatrixMatcher: public Catch::Matchers::Impl::MatcherImpl<ApproxMatrixMatcher, Eigen::MatrixXd> {
public:
	ApproxMatrixMatcher(const Eigen::MatrixXd& data,
			double epsilon = std::numeric_limits<float>::epsilon() * 100, double scale = 1.0);

	virtual ~ApproxMatrixMatcher();
	explicit ApproxMatrixMatcher(const ApproxMatrixMatcher& other);

	virtual bool match(Eigen::MatrixXd const& expr) const;
	virtual std::string toString() const;

private:
	Eigen::MatrixXd m_data;
	double m_epsilon, m_scale;
};

#endif /* APPROXMATRIXMATCHER_H_ */
