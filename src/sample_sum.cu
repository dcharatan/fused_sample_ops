#include "common.cuh"
#include "sample_sum.cuh"

int divide_round_up(const int a, const int b) {
  // This is integer division that rounds up. It answers the question "how many of b are
  // needed to hold a?"
  return (a - 1) / b + 1;
}

void fused_sample_ops::sample_sum_forward(torch::Tensor images,
                                          torch::Tensor samples,
                                          torch::Tensor weights,
                                          torch::Tensor outputs) {
  // We assume that 32-bit indexing can be used and that only float32 and float64 are
  // supported.
  const int B = weights.size(0);
  const int HD = weights.size(1);
  const int Q = weights.size(2);
  const int D = weights.size(3);
  const int C = images.size(1);

  // Compute the number of elements each thread initially loads and sums before any
  // reduction is carried out. This is two unless a value of two would not be enough to
  // sum across the D dimension using a single thread block.
  int initial_loads_per_thread = 2;
  if (BLOCK_SIZE * initial_loads_per_thread < D) {
    initial_loads_per_thread = divide_round_up(D, BLOCK_SIZE);
  }

  // Compute a padded version of D that is greater than or equal to D and cleanly
  // divisible by initial_loads_per_thread.
  const int D_padded =
      divide_round_up(D, initial_loads_per_thread) * initial_loads_per_thread;

  // Compute the number of threads needed for any given sum.
  const int threads_per_sum = D_padded / initial_loads_per_thread;

  // Compute the number of sums that can be carried out in each block.
  const int sums_per_block = BLOCK_SIZE / threads_per_sum;

  // Compute the total number of blocks needed to carry out the sums.
  const int sums_total = B * HD * Q * C;
  const int blocks_total = divide_round_up(sums_total, sums_per_block);

  if (blocks_total > 0) {
    AT_DISPATCH_FLOATING_TYPES(
        images.scalar_type(), "sample_sum_forward", ([&] {
          sample_sum_forward_kernel<scalar_t><<<blocks_total, BLOCK_SIZE>>>(
              sums_total, threads_per_sum, sums_per_block, D_padded,
              initial_loads_per_thread,
              images.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
              samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
              weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
              outputs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
        }));
  }
}

void fused_sample_ops::sample_sum_backward(torch::Tensor output_gradients,
                                           torch::Tensor images,
                                           torch::Tensor samples,
                                           torch::Tensor weights,
                                           torch::Tensor image_gradients,
                                           torch::Tensor sample_gradients,
                                           torch::Tensor weight_gradients) {
  // We assume that 32-bit indexing can be used and that only float32 and float64 are
  // supported. Note that we parallelize over different dimensions than in the forward
  // pass, since the backward pass has no dependency between depths.
  int B = weights.size(0);
  int Q = weights.size(2);
  int D = weights.size(3);
  int num_threads = B * Q * D;
  if (num_threads > 0) {
    AT_DISPATCH_FLOATING_TYPES(
        images.scalar_type(), "sample_sum_backward", ([&] {
          sample_sum_backward_kernel<scalar_t>
              <<<get_blocks(num_threads, BLOCK_SIZE), BLOCK_SIZE>>>(
                  num_threads,
                  output_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  images.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  image_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  sample_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  weight_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
        }));
  }
}