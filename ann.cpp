#include "ann.h"

using namespace std;

const char* FileNotFoundEx::what() const throw(){
    return "File not found";
}

const char* WrongANNVersionEx::what() const throw(){
    return "ANN input file incompatible with current build";
}

const char* CorruptANNFileEx::what() const throw(){
    return "ANN file corrupt";
}


ANN::ANN(const ANN& other) : hidden_nodes_(other.hidden_nodes_), output_nodes_(other.output_nodes_), hidden_output_(other.hidden_output_),
    output_(other.output_), input_range_(other.input_range_), output_range_(other.output_range_), input_norm_range_(other.input_norm_range_),
    output_norm_range_(other.output_norm_range_){
}

ANN::ANN(unsigned input, unsigned hidden, unsigned output) : hidden_nodes_(input, hidden + 1), output_nodes_(hidden + 1, output),
    hidden_output_(Eigen::MatrixXf::Zero(1, 1)), output_(Eigen::MatrixXf::Zero(1, 1)), input_range_(input, 2), output_range_(output, 2),
    input_norm_range_(input + 1, 2), output_norm_range_(output, 2){

    auto gen = bind(uniform_real_distribution<float>(-1.f, 1.f), mt19937(time(0)));

    for(int k = 0; k < hidden_nodes_.rows(); ++k){
        for(int i = 0; i < hidden_nodes_.cols() - 1; ++i){
            hidden_nodes_(k, i) = gen();
        }
    }

    hidden_nodes_.col(hidden_nodes_.cols() - 1) = Eigen::VectorXf::Zero(hidden_nodes_.rows());
    hidden_nodes_(hidden_nodes_.rows() - 1, hidden_nodes_.cols() - 1) = 1;

    for(int k = 0; k < output_nodes_.rows(); ++k){
        for(int i = 0; i < output_nodes_.cols(); ++i){
            output_nodes_(k, i) = gen();
        }
    }

    getNormalizedRange(input_range_, NORM_RANGE_MIN_IN, NORM_RANGE_MAX_IN);
    getNormalizedRange(input_norm_range_, NORM_RANGE_MIN_IN, NORM_RANGE_MAX_IN);
    getNormalizedRange(output_range_, NORM_RANGE_MIN_OUT, NORM_RANGE_MAX_OUT);
    getNormalizedRange(output_norm_range_, NORM_RANGE_MIN_OUT, NORM_RANGE_MAX_OUT);
}

ANN::ANN(string filename) : hidden_output_(Eigen::MatrixXf::Zero(1, 1)), output_(Eigen::MatrixXf::Zero(1, 1)){
    ifstream in_file(filename);
    if(!in_file)
        throw FileNotFoundEx();

    int input, hidden, output;

    in_file >> input >> hidden >> output;

    hidden_nodes_ = Eigen::MatrixXf(input, hidden);
    output_nodes_ = Eigen::MatrixXf(hidden, output);
    input_range_ = Eigen::MatrixXf(input, 2);
    output_range_ = Eigen::MatrixXf(output, 2);

    input_norm_range_ = Eigen::MatrixXf(input, 2);
    output_norm_range_ = Eigen::MatrixXf(output, 2);

    getNormalizedRange(input_norm_range_, NORM_RANGE_MIN_IN, NORM_RANGE_MAX_IN);
    getNormalizedRange(output_norm_range_, NORM_RANGE_MIN_OUT, NORM_RANGE_MAX_OUT);

    for(int k = 0; k < input; ++k){
        if(in_file.peek() == EOF)
            throw CorruptANNFileEx();

        in_file >> input_range_(k, 0);
        in_file >> input_range_(k, 1);

        assert(input_range_(k, 0) <= input_range_(k, 1));
    }

    for(int k = 0; k < output; ++k){
        if(in_file.peek() == EOF)
            throw CorruptANNFileEx();

        in_file >> output_range_(k, 0);
        in_file >> output_range_(k, 1);

        assert(output_range_(k, 0) < output_range_(k, 1));
    }

    for(int k = 0; k < input; ++k){
        for(int i = 0; i < hidden; ++i){
            if(in_file.peek() == EOF)
                throw CorruptANNFileEx();

            in_file >> hidden_nodes_(k, i);
        }
    }

    for(int k = 0; k < hidden; ++k){
        for(int i = 0; i < output; ++i){
            if(in_file.peek() == EOF)
                throw CorruptANNFileEx();

            in_file >> output_nodes_(k, i);
        }
    }

    in_file.close();
}

ANN::~ANN(){

}

void ANN::iterate(const Eigen::MatrixXf& input){
    if(input.cols() != hidden_nodes_.rows())
        throw WrongANNVersionEx();

    Eigen::MatrixXf normalized_input = rescale(input, input_range_, input_norm_range_);

    if(!isMatEq(hidden_output_, Eigen::MatrixXf::Zero(1, 1))){
        output_ = hidden_output_ * output_nodes_;
    }

    hidden_output_ = normalized_input * hidden_nodes_;

    for(int k = 0; k < output_.rows(); ++k){
        for(int i = 0; i < output_.cols(); ++i){
            output_(k, i) = sigmoid(output_(k, i));
        }
    }

    for(int k = 0; k < hidden_output_.rows(); ++k){
        for(int i = 0; i < hidden_output_.cols() - 1; ++i){
            hidden_output_(k, i) = sigmoid(hidden_output_(k, i));
        }
    }
}

Eigen::MatrixXf ANN::fullSingleRun(const Eigen::MatrixXf& input){
    if(input.cols() != hidden_nodes_.rows())
        throw WrongANNVersionEx();

    Eigen::MatrixXf hidden_out;
    Eigen::MatrixXf normalized_inputs = rescale(input, input_range_, input_norm_range_);

    forwardEval(normalized_inputs, hidden_out, output_);

    //cout << output_ << endl;

    return rescale(output_, output_norm_range_, output_range_);
}

Eigen::MatrixXf ANN::getLastOutput(){
    return rescale(output_, output_norm_range_, output_range_);
}

void ANN::save(string filename){
    ofstream of;
    of.open(filename);

    of << hidden_nodes_.rows() << " " << output_nodes_.rows() << " " << output_nodes_.cols();

    for(int k = 0; k < input_range_.rows(); ++k){
        of << " " << input_range_(k, 0);
        of << " " << input_range_(k, 1);
    }

    for(int k = 0; k < output_range_.rows(); ++k){
        of << " " << output_range_(k, 0);
        of << " " << output_range_(k, 1);
    }

    for(int k = 0; k < hidden_nodes_.rows(); ++k){
        for(int i = 0; i < hidden_nodes_.cols(); ++i){
            of << " " << hidden_nodes_(k, i);
        }
    }

    for(int k = 0; k < output_nodes_.rows(); ++k){
        for(int i = 0; i < output_nodes_.cols(); ++i){
            of << " " << output_nodes_(k, i);
        }
    }

    of.close();
}

void ANN::forwardEval(const Eigen::MatrixXf& input, Eigen::MatrixXf& hidden_out, Eigen::MatrixXf& out){
    hidden_out = input * hidden_nodes_;
    for(int k = 0; k < hidden_out.rows(); ++k){
        for(int i = 0; i < hidden_out.cols() - 1; ++i){
            hidden_out(k, i) = sigmoid(hidden_out(k, i));
        }
    }

    out = hidden_out * output_nodes_;
    for(int k = 0; k < out.rows(); ++k){
        for(int i = 0; i < out.cols(); ++i){
            out(k, i) = sigmoid(out(k, i));
        }
    }
}

Eigen::MatrixXf ANN::trainSingular(const Eigen::MatrixXf& training_sample, const Eigen::MatrixXf& expected_out){
    assert(training_sample.rows() == 1);

    Eigen::MatrixXf hidden_out, out;

    forwardEval(training_sample, hidden_out, out);

    assert(out.rows() == expected_out.rows() && out.cols() == expected_out.cols());

    Eigen::MatrixXf out_error(out.rows(), out.cols());
    Eigen::MatrixXf actual_error(out.rows(), out.cols());
    for(int k = 0; k < out.rows(); ++k){
        for(int i = 0; i < out.cols(); ++i){
            out_error(k, i) = (out(k, i) * (1.f - out(k, i))) * (expected_out(k, i) - out(k, i));
            actual_error(k, i) = expected_out(k, i) - out(k, i);
        }
    }
    out_error.transposeInPlace();

    Eigen::MatrixXf hidden_error = output_nodes_ * out_error;

    //drop last row of error as it is for bias and not relevant for hidden nodes
    Eigen::MatrixXf hidden_error_nb = hidden_error.block(0, 0, hidden_error.rows() - 1, hidden_error.cols());

    for(int k = 0; k < hidden_out.rows(); ++k){
        for(int i = 0; i < hidden_out.cols() - 1; ++i){
            hidden_error_nb(i, k) *= hidden_out(k, i) * (1.f - hidden_out(k, i));
        }
    }

    Eigen::MatrixXf out_corrections = out_error * hidden_out;
    Eigen::MatrixXf hidden_corrections = hidden_error_nb * training_sample;

    //reversed rows and columns on hidden_corrections as it is row instead of column major
    for(int k = 0; k < output_nodes_.rows(); ++k){
        for(int i = 0; i < output_nodes_.cols(); ++i){
            output_nodes_(k, i) += LEARNING_RATE * out_corrections(i, k);
        }
    }

    for(int k = 0; k < hidden_nodes_.rows(); ++k){
        for(int i = 0; i < hidden_nodes_.cols() - 1; ++i){
            hidden_nodes_(k, i) += LEARNING_RATE * hidden_corrections(i, k);
        }
    }

    return actual_error;
}

void ANN::trainMultiple(const Eigen::MatrixXf& training_samples, const Eigen::MatrixXf& expected_out){
    assert(training_samples.rows() == expected_out.rows());
    assert(training_samples.cols() == hidden_nodes_.rows());

    unsigned counter = 0;
    auto gen = bind(uniform_int_distribution<int>(0, training_samples.rows() - 1), mt19937(time(0)));

    Eigen::MatrixXf norm_train_samples = rescale(training_samples, input_range_, input_norm_range_);
    Eigen::MatrixXf norm_exp_out = rescale(expected_out, output_range_, output_norm_range_);

    while(true){
        float acc_error = 0.f;
        for(int k = 0; k < norm_train_samples.rows(); ++k){
            int pos = gen();
            Eigen::MatrixXf error = trainSingular(norm_train_samples.row(pos), norm_exp_out.row(pos));

            for(int i = 0; i < error.cols(); ++i){
                acc_error += fabs(error(0, i));
            }

            if((k%1000000 == 0) && k != 0){
                cout << k << " average err: " << acc_error/1000000.f << endl;
                acc_error = 0.f;
            }
        }

        counter++;

        if(counter >= LIMIT || acc_error < (float)norm_train_samples.rows() * LEARNING_EPSILON){
            cout << "finished training" << endl;
            return;
        }


    }
}

float ANN::sigmoid(float signal){
    return 1.f/(1.f + exp(-signal));
}

Eigen::MatrixXf ANN::rescale(const Eigen::MatrixXf& input, const Eigen::MatrixXf& old_range, const Eigen::MatrixXf& new_range){
    Eigen::MatrixXf rescaled_input(input.rows(), input.cols());

    for(int k = 0; k < input.rows(); ++k){
        for(int i = 0; i < input.cols(); ++i){

            if((old_range(i, 1) - old_range(i, 0)) == 0){
                rescaled_input(k, i) = (new_range(i, 1) - new_range(i, 0)) / 2;
            }
            else{
                float temp = (input(k, i) - old_range(i, 0)) / (old_range(i, 1) - old_range(i, 0));
                rescaled_input(k, i) = (temp * (new_range(i, 1) - new_range(i, 0))) + new_range(i, 0);
            }
        }
    }

    return rescaled_input;
}

void ANN::getNormalizedRange(Eigen::MatrixXf& range, float min, float max){
    assert(range.cols() == 2);

    for(int k = 0; k < range.rows(); ++k){
        range(k, 0) = min;
        range(k, 1) = max;
    }
}

void ANN::setRanges(const Eigen::MatrixXf& input_range, const Eigen::MatrixXf& output_range){
    input_range_ = input_range;
    output_range_ = output_range;
}

bool ANN::isMatEq(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b){
    if(a.rows() == b.rows() && a.cols() == b.cols()){
        for(int k = 0; k < a.rows(); ++k){
            for(int i = 0; i < a.cols(); ++i){
                if(a(k, i) != b(k, i)){
                    return false;
                }
            }
        }

        return true;
    }

    return false;
}
