#!/usr/bin/env python3
"""Compile the actual CPU reference/error routines; this does not test CUDA."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def routine(path, name):
    text = (ROOT / path).read_text()
    start = text.rfind('\n', 0, text.index(name)) + 1
    body = text.index('{', start)
    depth = 1
    end = body + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


source = '''#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>
#define REQUIRE(condition) do { if (!(condition)) { \\
    std::fprintf(stderr, "check failed at line %d: %s\\n", __LINE__, #condition); \\
    return EXIT_FAILURE; } } while (0)
'''
for path, names in [
    ('3-ger/ger.cu', ['ger_cpu', 'compute_error_norm']),
    ('4-gemm/gemm.cu', ['batched_gemm_cpu', 'forward_error',
                        'max_batch_forward_error']),
]:
    source += '\n'.join(routine(path, name) for name in names) + '\n'

source += r'''
int main() {
    // Rectangular GER, nonzero initial A; arrays below are column-major.
    float x[] = {1, -2}, y[] = {3, 0, -1};
    float a[] = {1, 2, 3, 4, 5, 6};
    const float expected[] = {7, -10, 3, 4, 3, 10};
    ger_cpu(2, 3, 2, x, y, a);
    REQUIRE(compute_error_norm(6, a, expected) == 0);
    ger_cpu(2, 3, 0, x, y, a);
    REQUIRE(compute_error_norm(6, a, expected) == 0);

    // Two distinct nonsymmetric products catch transposition/batch mistakes.
    const float A[] = {1, 3, 2, 4, 2, 0, -1, 3};
    const float B[] = {5, 7, 6, 8, 4, 1, 2, -2};
    const float product[] = {19, 43, 22, 50, 7, 3, 6, -6};
    float C[8];
    batched_gemm_cpu(2, 2, A, B, C);
    REQUIRE(max_batch_forward_error(2, 2, C, product) == 0);

    // Every supported n, multiple batches: left/right identity and zero.
    for (int n = 1; n <= 32; ++n) {
        const int total = 3*n*n;
        std::vector<float> identity(total, 0), data(total), out(total), zero(total, 0);
        for (int i = 0; i < total; ++i) data[i] = (i % 23) - 11;
        for (int b = 0; b < 3; ++b)
            for (int i = 0; i < n; ++i) identity[b*n*n+i*n+i] = 1;
        batched_gemm_cpu(n, 3, data.data(), identity.data(), out.data());
        REQUIRE(out == data);
        batched_gemm_cpu(n, 3, identity.data(), data.data(), out.data());
        REQUIRE(out == data);
        batched_gemm_cpu(n, 3, zero.data(), data.data(), out.data());
        REQUIRE(out == zero);
    }
    float zero[] = {0, 0, 0, 0}, one[] = {1, 0, 0, 0};
    REQUIRE(compute_error_norm(4, zero, zero) == 0);
    REQUIRE(std::isinf(compute_error_norm(4, one, zero)));
    REQUIRE(forward_error(2, zero, zero) == 0);
    REQUIRE(std::isinf(forward_error(2, one, zero)));
    const float ref[] = {1, 1, 1, 1}, changed[] = {3, 1, 1, 1};
    REQUIRE(std::fabs(compute_error_norm(4, changed, ref) - 1.0) < 1e-15);
    REQUIRE(std::fabs(forward_error(2, changed, ref) - 0.5) < 1e-15);
    float bad[] = {NAN, 0, 0, 0};
    REQUIRE(std::isinf(compute_error_norm(4, bad, ref)));
    REQUIRE(std::isinf(max_batch_forward_error(2, 1, bad, ref)));
    bad[0] = INFINITY;
    REQUIRE(std::isinf(max_batch_forward_error(2, 1, bad, ref)));
    std::puts("PASS: CPU references and error metrics (GEMM n=1..32, 3 batches).");
    std::puts("CUDA kernels, cuBLAS, launches, and GPU performance remain untested.");
}
'''
with tempfile.TemporaryDirectory(prefix='session4-cpu-') as temporary:
    cpp = Path(temporary) / 'checks.cpp'
    executable = Path(temporary) / 'checks'
    cpp.write_text(source)
    command = shlex.split(os.environ.get('CXX', 'c++')) + [
        '-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror',
        '-fsanitize=address,undefined', str(cpp), '-o', str(executable)]
    print('Compiler:', subprocess.check_output(command[:1] + ['--version'], text=True).splitlines()[0], flush=True)
    print('Flags: -std=c++11 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined', flush=True)
    subprocess.run(command, check=True)
    subprocess.run([str(executable)], check=True)
