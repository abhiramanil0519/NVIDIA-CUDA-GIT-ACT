#include iostream

__global__ void add(int a, int b, int c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int N = 8;
    int a[N], b[N], c[N];

    for(int i=0;iN;i++)
    {
        a[i] = i;
        b[i] = ii;
    }

    int d_a, d_b, d_c;

    cudaMalloc((void)&d_a, Nsizeof(int));
    cudaMalloc((void)&d_b, Nsizeof(int));
    cudaMalloc((void)&d_c, Nsizeof(int));

    cudaMemcpy(d_a, a, Nsizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, Nsizeof(int), cudaMemcpyHostToDevice);

    add1, N(d_a, d_b, d_c);

    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, Nsizeof(int), cudaMemcpyDeviceToHost);

    stdcout  Resultn;
    for(int i=0;iN;i++)
        stdcout  a[i]   +   b[i]   =   c[i]  n;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
