#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
#define EXTERNC extern "C" __declspec(dllexport)
#else
#define EXTERNC
#endif

#define THREADS 192

typedef struct Complex {
    double re, im;
} Complex_t;

EXTERNC void iterate(int max_iters);

EXTERNC void draw();

EXTERNC void dispose();

EXTERNC uint8_t *setupCanvas(int fWidth, int fHeight, double re1, double re2, double im1, double im2);