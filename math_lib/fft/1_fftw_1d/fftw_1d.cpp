#include <vector>
#include <iostream>
#include <math.h>
#include <fftw3.h>

int main()
{
    const size_t Num = 16;
    fftwf_complex *in, *out;
    fftwf_plan p;

    in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Num);
    out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Num);

    for (size_t i = 0; i < Num; i++) {
        in[i][0] = i;
        in[i][1] = 0;
    }

    p = fftwf_plan_dft_1d(Num, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    std::cout << "Output:\n";
    for (size_t i = 0; i < Num; i++)
        std::cout << out[i][0] << ", " << out[i][1] << "i\n";


    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    return 0;
}
