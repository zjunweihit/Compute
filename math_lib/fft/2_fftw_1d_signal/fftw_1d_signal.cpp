#include <vector>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <fftw3.h>

/*
 * Sample frequency fs = 256 Hz
 * Sample number N = 256
 *
 * Signal 2+3*sin(2*PI*50*t) + 5*cos(2*PI*75*t + PI*30/180)
 *   sin() = cos() moves right 90,
 * Signal frequency 0, 50, 75
 *
 * Each point of FFT
 *   point = a + bi
 *   Magnitude An = sqrt(a*a + b*b) = A * N / 2
 *     => A(signal) = An * 2 / N
 *     => A(DC) = An / N
 *   Phase Pn = atan2(b, a)
 *
 *   fn = fs * i / N
 *   interval of fn = fs / N
 *
 */
int main()
{
    const size_t N = 256;
    const double fs = 256;
    const double PI = 3.14159265;
    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    for (size_t i = 0; i < N; i++) {
        in[i][0] = 2.0 + 3 * sin((double)2*PI*50*i/N) + 5*cos((double)2*PI*75*i/N + PI*90/180);
        in[i][1] = 0;
    }

    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    std::cout << "Output:\n";
    for (size_t i = 0; i < N; i++) {
        double fn = (float)i * fs / N;
        double An = sqrt(pow(out[i][0], 2) + pow(out[i][1], 2));
        double A;
        if (i == 0)
            A = An / N;
        else
            A = An * 2 / N;
        double Pn = atan2(out[i][1], out[i][0]) * 180 / PI;

        std::cout << "[" << std::setw(4) << i << "] " <<
            "fn: " << std::setw(4) << fn << ", " <<
            "A: " << std::setw(12) << A << ", " <<
            "Pn: " << std::setw(8) << Pn <<  "\n";
        //std::cout << out[i][0] << ", " << out[i][1] << "i\n";
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return 0;
}
