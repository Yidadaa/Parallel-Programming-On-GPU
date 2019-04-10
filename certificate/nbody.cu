#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Do not modify this function. A constraint of this exercise is
 * that it remain a host function.
 */

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 */

__global__
void bodyForce(Body *p, Body *tmp, float dt, int n) {
  int xindex = threadIdx.x + blockIdx.x * blockDim.x;
  int xstride = blockDim.x * gridDim.x;

  for(int i = xindex; i < n; i += xstride)
  {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;

      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }
    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
    tmp[i].vx = p[i].vx;
    tmp[i].vy = p[i].vy;
    tmp[i].vz = p[i].vz;
    tmp[i].x = p[i].x + p[i].vx*dt;
    tmp[i].y = p[i].y + p[i].vy*dt;
    tmp[i].z = p[i].z + p[i].vz*dt;
  }
}

int main(const int argc, const char** argv) {

  int deviceId;
  int numberOfSMs;
  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  
  size_t nthreads = 512;
  size_t nblocks = 32 * numberOfSMs;
  
  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  /*
   * Do not change the value for `nBodies` here. If you would like to modify it,
   * pass values into the command line.
   */

  int nBodies = 2<<11;
  int salt = 0;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  /*
   * This salt is for assessment reasons. Tampering with it will result in automatic failure.
   */

  if (argc > 2) salt = atoi(argv[2]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies * sizeof(Body);
  
  float *host_buf;
  float *device_buf_1;
  float *device_buf_2;
  
  cudaMalloc(&device_buf_1, bytes);
  cudaMalloc(&device_buf_2, bytes);
  cudaMallocHost(&host_buf, bytes);

  /*
   * As a constraint of this exercise, `randomizeBodies` must remain a host function.
   */

  randomizeBodies(host_buf, 6 * nBodies); // Init pos / vel data
  
  cudaMemcpy(device_buf_1, host_buf, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_buf_2, host_buf, bytes, cudaMemcpyHostToDevice);

  double totalTime = 0.0;
  
  int flag = 0;

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  /*******************************************************************/
  // Do not modify these 2 lines of code.
  for (int iter = 0; iter < nIters; iter++) {
      StartTimer();
      /*******************************************************************/

      /*
       * You will likely wish to refactor the work being done in `bodyForce`,
       * as well as the work to integrate the positions.
       */
      flag = 1 - flag;
      if (flag) {
          bodyForce<<<nblocks, nthreads>>>((Body*)device_buf_1, (Body*)device_buf_2, dt, nBodies);
      } else {
          bodyForce<<<nblocks, nthreads>>>((Body*)device_buf_2, (Body*)device_buf_1, dt, nBodies);
      }

      /*******************************************************************/
      // Do not modify the code in this section.
      const double tElapsed = GetTimer() / 1000.0;
      totalTime += tElapsed;
  }
  if (flag) {
      cudaMemcpy(host_buf, device_buf_2, bytes, cudaMemcpyHostToDevice);
  } else {
      cudaMemcpy(host_buf, device_buf_1, bytes, cudaMemcpyHostToDevice);
  }
  
  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

#ifdef ASSESS
  checkPerformance(host_buf, billionsOfOpsPerSecond, salt);
#else
  checkAccuracy(host_buf, nBodies);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
  salt += 1;
#endif
  /*******************************************************************/

  /*
   * Feel free to modify code below.
   */

  cudaFree(device_buf_1);
  cudaFree(device_buf_2);
  cudaFreeHost(host_buf);
}
