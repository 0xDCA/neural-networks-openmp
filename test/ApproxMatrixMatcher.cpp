#include "ApproxMatrixMatcher.h"

#include <sstream>

ApproxMatrixMatcher::ApproxMatrixMatcher(const Eigen::MatrixXd& data, double epsilon, double scale) :
		m_data(data), m_epsilon(epsilon), m_scale(scale) {

}

ApproxMatrixMatcher::ApproxMatrixMatcher(const ApproxMatrixMatcher& other) :
		m_data(other.m_data), m_epsilon(other.m_epsilon), m_scale(other.m_scale) {

}

ApproxMatrixMatcher::~ApproxMatrixMatcher() {

}

bool ApproxMatrixMatcher::match(Eigen::MatrixXd const& expr) const {
	if (expr.cols() != m_data.cols() || expr.rows() != m_data.rows()) {
		return false;
	}

	for (int i = 0; i < m_data.cols(); ++i) {
		for(int j = 0; j < m_data.rows(); ++j) {
			if (expr(j, i) != Approx(m_data(j, i)).epsilon(m_epsilon).scale(m_scale)) {
				return false;
			}
		}
	}

	return true;
}

std::string ApproxMatrixMatcher::toString() const {
	std::ostringstream oss;

	oss << "Approx: \n";
	oss << m_data;
	oss << '\n';
	oss << "scale = " << m_scale << ", epsilon = " << m_epsilon << '\n';

	return oss.str();
}
