#include "../header/wavelib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  // 变量声明
  int i, N, J, subscale, a0, iter, nd, k;
  double *inp, *oup;
  double dt, dj, s0, param, mn;
  double td, tn, den, num, recon_mean, recon_var;
  double mod = 0.0;
  cwt_object wt;

  struct timeval start, end;
  long seconds, useconds;
  double total_time;

  FILE *ifp; // 输入文件指针
  FILE *fp;  // 输出文件指针
  double temp[1200]; // 临时存储输入数据的数组

  char *wave = "morlet"; // 使用Morlet小波，可选"paul"和"dog"
  char *type = "pow";

  // 初始化参数
  N = 504;
  param = 6.0;
  subscale = 2;
  dt = 0.25;
  s0 = dt;
  dj = 1.0 / (double)subscale;
  J = 11 * subscale;   // 总的尺度数
  a0 = 2;            // 幂次

  // 打开输入文件
  ifp = fopen("/home/zdhjs-05/myGitHubCode/wavelib/test/sst_nino3.dat", "r");
  i = 0;
  if (!ifp) {
    printf("无法打开文件");
    exit(100);
  }
  while (!feof(ifp)) {
    fscanf(ifp, "%lf \n", &temp[i]);
    i++;
  }

  // 关闭输入文件
  fclose(ifp);

  // 初始化小波变换对象
  wt = cwt_init(wave, param, N, dt, J);

  // 分配内存
  inp = (double *)malloc(sizeof(double) * N);
  oup = (double *)malloc(sizeof(double) * N);

  // 将数据从临时数组复制到输入数组
  for (i = 0; i < N; ++i) {
    inp[i] = temp[i];
  }

  // 设置小波变换的尺度
  setCWTScales(wt, s0, dj, type, a0);

  // 记录开始时间
  gettimeofday(&start, NULL);

  // 执行连续小波变换
  cwt(wt, inp);

  // 记录结束时间
  gettimeofday(&end, NULL);

  // 计算总时间
  seconds  = end.tv_sec  - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;
  total_time = seconds + useconds/1E6;
  printf("程序耗时: %f 秒\n", total_time);

  // 输出平均值
  printf("\n MEAN %g \n", wt->smean);

  // 计算输出信号的平均模
  mn = 0.0;
  for (i = 0; i < N; ++i) {
    mn += sqrt(wt->output[i].re * wt->output[i].re + wt->output[i].im * wt->output[i].im);
  }

  // 输出小波变换的总结
  cwt_summary(wt);

  printf("\n abs mean %g \n", mn / N);
  printf("\n\n");
  printf("Let CWT w = w(j, n/2 - 1) where n = %d\n\n", N);
  // nd = N / 2 - 1;

  printf("%-15s%-15s%-15s%-15s \n", "j", "Scale", "Period", "ABS(w)^2");

  // 保存结果数据到文件
  fp = fopen("cwt_result.dat", "w");
  for (nd = 0; nd < N; ++nd) {
    for (k = 0; k < wt->J; ++k) {
      iter = k + nd * wt->J;
      mod = wt->output[iter].re * wt->output[iter].re +
            wt->output[iter].im * wt->output[iter].im;
      fprintf(fp, "%.6f %.6f %.6f\n", wt->scale[k], wt->period[k], mod);
    }
  }

  // 执行逆小波变换
  icwt(wt, oup);

  // 计算重构误差
  num = den = recon_var = recon_mean = 0.0;
  printf("\n\n");
  printf("Signal Reconstruction\n");
  printf("%-15s%-15s%-15s \n", "i", "Input(i)", "Output(i)");

  // 打印最后10个数据点的输入和输出
  for (i = N - 10; i < N; ++i) {
    printf("%-15d%-15lf%-15lf \n", i, inp[i], oup[i]);
  }

  // 计算重构的均值和方差
  for (i = 0; i < N; ++i) {
    td = inp[i];
    tn = oup[i] - td;
    num += (tn * tn);
    den += (td * td);
    recon_mean += oup[i];
  }

  recon_var = sqrt(num / N);
  recon_mean /= N;

  printf("\nRMS Error %g \n", sqrt(num) / sqrt(den));
  printf("\nVariance %g \n", recon_var);
  printf("\nMean %g \n", recon_mean);

  // 释放内存并清理
  free(inp);
  free(oup);
  cwt_free(wt);
  return 0;
}