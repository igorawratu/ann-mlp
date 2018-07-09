#ifndef ANN_H_
#define ANN_H_

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <exception>
#include <random>
#include <iostream>

const int HIDDEN_NODES = 20;
const float LEARNING_RATE = 0.01f;
const float LEARNING_EPSILON = 0.001f;
const unsigned LIMIT = 1;
const float NORM_RANGE_MIN_IN = -1.f;
const float NORM_RANGE_MAX_IN = 1.f;
const float NORM_RANGE_MIN_OUT = 0.1f;
const float NORM_RANGE_MAX_OUT = 0.9f;

class FileNotFoundEx : public std::exception{
    virtual const char* what() const throw();
};

class WrongANNVersionEx: public std::exception{
    virtual const char* what() const throw();
};

class CorruptANNFileEx: public std::exception{
    virtual const char* what() const throw();
};

class ANN{
friend class TestANN;

private:
    Eigen::MatrixXf hidden_nodes_;
    Eigen::MatrixXf output_nodes_;
    Eigen::MatrixXf hidden_output_;
    Eigen::MatrixXf output_;

    Eigen::MatrixXf input_range_;
    Eigen::MatrixXf output_range_;
    Eigen::MatrixXf input_norm_range_;
    Eigen::MatrixXf output_norm_range_;

public:
    ANN() = delete;
    ANN(const ANN& other);
    ANN(unsigned input, unsigned hidden, unsigned output);
    ANN(std::string filename);
    ~ANN();

    void iterate(const Eigen::MatrixXf& input);
    Eigen::MatrixXf fullSingleRun(const Eigen::MatrixXf& input);
    Eigen::MatrixXf getLastOutput();
    void save(std::string filename);
    void setRanges(const Eigen::MatrixXf& input_range, const Eigen::MatrixXf& output_range);

    void trainMultiple(const Eigen::MatrixXf& training_samples, const Eigen::MatrixXf& expected_out);

private:
    void forwardEval(const Eigen::MatrixXf& input, Eigen::MatrixXf& hidden_out, Eigen::MatrixXf& out);
    float sigmoid(float signal);
    Eigen::MatrixXf trainSingular(const Eigen::MatrixXf& training_sample, const Eigen::MatrixXf& expected_out);
    Eigen::MatrixXf rescale(const Eigen::MatrixXf& input, const Eigen::MatrixXf& old_range, const Eigen::MatrixXf& new_range);
    void getNormalizedRange(Eigen::MatrixXf& range, float min, float max);
    bool isMatEq(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b);
};

#endif

