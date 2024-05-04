#ifndef LSTM_H
#define LSTM_H

typedef struct
{
    double *wi;
    double *wf;
    double *wg;
    double *wo;
    double *bi;
    double *bf;
    double *bg;
    double *bo;
    double *whi;
    double *whf;
    double *whg;
    double *who;
    double *bhi;
    double *bhf;
    double *bhg;
    double *bho;
    double *fc_w;
    double *fc_b;

    int hidden_dim, input_dim, num_layers, num_labels;

} LSTM_t;

void lstm_init(LSTM_t *lstm);
double sigmoid(double x);
double tanh_activation(double x);
void softmax(double *input, int length);
void matrix_vector_multiply(double *matrix, double *vector, double *result, int rows, int cols);
void vector_add(double *vector1, double *vector2, int length);
void matrix_multiply_and_add(double *input,
                             double *w1, double *b1, double *w2, double *b2,
                             double rows, double cols,
                             double *result);
void calculate_i(double *input,
                 double *w, double *b, double *wh, double *bh,
                 double rows, double cols,
                 double *result);
void calculate_f(double *input,
                 double *w, double *b, double *wh, double *bh,
                 double rows, double cols,
                 double *result);
void calculate_g(double *input,
                 double *w, double *b, double *wh, double *bh,
                 double rows, double cols,
                 double *result);
void calculate_o(double *input,
                 double *w, double *b, double *wh, double *bh,
                 double rows, double cols,
                 double *result);
void fullconncetlayer_forward(double *input, LSTM_t *lstm, double *result);
void lstm_forward_step(double* input, LSTM_t* lstm);

#endif