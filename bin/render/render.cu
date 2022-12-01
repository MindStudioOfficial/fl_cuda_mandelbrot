#include <render.cuh>

int w, h;
double r1, r2, i1, i2;
bool setup = false;
uint64_t *d_iters = nullptr;
Complex_t *d_vals = nullptr;
uint8_t *img = nullptr, *d_img = nullptr;

uint64_t maxiter = 0;

__device__ Complex_t operator+(Complex_t a, Complex_t b)
{
    Complex_t c = {a.re + b.re, a.im + b.im};
    return c;
}

__device__ Complex_t operator*(Complex_t a, Complex_t b)
{
    Complex_t c = {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
    return c;
}

__device__ double cabs(Complex_t c)
{
    return sqrt(c.re * c.re + c.im * c.im);
}

__device__ uchar3 hsvToRGB(float3 hsv)
{
    float f = hsv.x / 60.0f;
    float hi = floorf(f);
    f = f - hi;
    float p = hsv.z * (1 - hsv.y);
    float q = hsv.z * (1 - hsv.y * f);
    float t = hsv.z * (1 - hsv.y * (1 - f));

    float r, g, b;

    if (hi == 0.0f || hi == 6.0f)
    {
        r = hsv.z;
        g = t;
        b = p;
    }
    else if (hi == 1.0f)
    {
        r = q;
        g = hsv.z;
        b = p;
    }
    else if (hi == 2.0f)
    {
        r = p;
        g = hsv.z;
        b = t;
    }
    else if (hi == 3.0f)
    {
        r = p;
        g = q;
        b = hsv.z;
    }
    else if (hi == 4.0f)
    {
        r = t;
        g = p;
        b = hsv.z;
    }
    else
    {
        r = hsv.z;
        g = p;
        b = q;
    }

    unsigned char red = __float2uint_rn(255.0f * r);
    unsigned char green = __float2uint_rn(255.0f * g);
    unsigned char blue = __float2uint_rn(255.0f * b);

    return make_uchar3(red, green, blue);
}

__global__ void kernelIterate(int w, int h, double r1, double r2, double i1, double i2, uint64_t *d_iters, Complex_t *d_vals)
{

    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= w * h)
        return;

    unsigned int px = pix % w;
    unsigned int py = h - (int)floorf(pix / (float)w);

    double real = r1 + ((r2 - r1) / w) * px;
    double imag = i1 + ((i2 - i1) / h) * py;

    Complex_t z = d_vals[pix];
    if (cabs(z) > 1000)
        return;
    Complex_t c = {real, imag};
    d_vals[pix] = z * z + c;
    d_iters[pix]++;
}

__global__ void kernelDraw(int width, int height, uint64_t *d_iters, uint8_t *d_img, uint64_t maxiter)
{
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= width * height)
        return;

    int64_t i = d_iters[pix];

    float h = (i % 360);
    float s = 1.f;
    float v = 1.f;

    if (i >= maxiter - 1)
        v = 0.f;

    uchar3 rgb = hsvToRGB(make_float3(h, s, v));

    d_img[pix * 4 + 3] = 255;
    d_img[pix * 4 + 0] = rgb.x;
    d_img[pix * 4 + 1] = rgb.y;
    d_img[pix * 4 + 2] = rgb.z;
}

EXTERNC void iterate()
{
    if (!setup)
    {
        printf("call setupCanvas before iterating!\n");
        return;
    }

    int pixs = w * h;

    int bw = THREADS;
    dim3 bs = dim3(bw);
    dim3 gs = dim3((unsigned int)ceilf(pixs / (float)bw));

    kernelIterate<<<gs, bs>>>(w, h, r1, r2, i1, i2, d_iters, d_vals);

    cudaDeviceSynchronize();

    maxiter++;
}

EXTERNC void draw()
{
    if (!setup)
    {
        printf("call setupCanvas before drawing!\n");
        return;
    }

    int pixs = w * h;
    int bw = THREADS;
    dim3 bs = dim3(bw);
    dim3 gs = dim3((unsigned int)ceilf(pixs / (float)bw));

    kernelDraw<<<gs, bs>>>(w, h, d_iters, d_img, maxiter);

    cudaDeviceSynchronize();

    size_t imgsize = sizeof(uint8_t) * pixs * 4;

    cudaMemcpy(img, d_img, imgsize, cudaMemcpyDeviceToHost);
}

EXTERNC void dispose()
{
    if (d_iters != nullptr)
    {
        cudaFree(d_iters);
        d_iters = nullptr;
    }
    if (d_vals != nullptr)
    {
        cudaFree(d_vals);
        d_vals = nullptr;
    }
    if (d_img != nullptr)
    {
        cudaFree(d_img);
        d_img = nullptr;
    }
    if (img != nullptr)
    {
        free(img);
        img = nullptr;
    }
    setup = false;
}

EXTERNC uint8_t *setupCanvas(int fWidth, int fHeight, double re1, double re2, double im1, double im2)
{
    dispose();
    w = fWidth;
    h = fHeight;
    r1 = re1;
    r2 = re2;
    i1 = im1;
    i2 = im2;

    maxiter = 0;

    size_t itersize = sizeof(uint64_t) * w * h;

    cudaMalloc(&d_iters, itersize);
    cudaMemset(d_iters, 0, itersize);

    size_t imgsize = sizeof(uint8_t) * w * h * 4;

    cudaMalloc(&d_img, imgsize);
    cudaMemset(d_img, 0, imgsize);

    img = (uint8_t *)calloc(1, imgsize);

    size_t valssize = sizeof(Complex_t) * w * h;

    cudaMalloc(&d_vals, valssize);
    cudaMemset(d_vals, 0, valssize);

    setup = true;

    return img;
}
