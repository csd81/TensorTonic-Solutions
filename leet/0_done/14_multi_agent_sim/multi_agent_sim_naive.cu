#include <cuda_runtime.h>

// Naive reference implementation of one boids neighbourhood step: for every
// agent, average the velocities of the agents within radius 5 (skipping the
// agent itself) and move it by 0.05 of the difference.
//
// One GPU thread per agent.  Each thread walks all N agents with direct
// global-memory scalar loads (no float4), computes the squared distance and
// accumulates.  No shared-memory tiling, no spatial hash, no self-count
// trick, no warp primitives.  Correctness-first baseline.

__global__ void boids_naive_kernel(const float* __restrict__ agents,
                                   float* __restrict__ agents_next, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const float xi = agents[4 * i + 0];
  const float yi = agents[4 * i + 1];
  const float vxi = agents[4 * i + 2];
  const float vyi = agents[4 * i + 3];

  float sx = 0.0f;
  float sy = 0.0f;
  int cnt = 0;
  for (int j = 0; j < N; ++j) {
    if (j == i) continue;  // an agent is not its own neighbour
    const float dx = xi - agents[4 * j + 0];
    const float dy = yi - agents[4 * j + 1];
    if (dx * dx + dy * dy < 25.0f) {
      sx += agents[4 * j + 2];
      sy += agents[4 * j + 3];
      ++cnt;
    }
  }

  float avx = vxi;
  float avy = vyi;
  if (cnt > 0) {
    avx = sx / static_cast<float>(cnt);
    avy = sy / static_cast<float>(cnt);
  }
  const float nvx = vxi + 0.05f * (avx - vxi);
  const float nvy = vyi + 0.05f * (avy - vyi);

  agents_next[4 * i + 0] = xi + nvx;
  agents_next[4 * i + 1] = yi + nvy;
  agents_next[4 * i + 2] = nvx;
  agents_next[4 * i + 3] = nvy;
}

// agents, agents_next are device pointers, 4 floats per agent.
extern "C" void solve(const float* agents, float* agents_next, int N) {
  if (N <= 0) return;
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  boids_naive_kernel<<<blocks, threads>>>(agents, agents_next, N);
}
