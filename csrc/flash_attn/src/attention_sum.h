#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "philox.cuh"
#include "utils.h"

namespace flash {

using namespace cute;


template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum_aw(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    AbsSumOp<float> abs_sum_op;   // 绝对值求和，其中包含inf的值会被置0
    thread_reduce_<zero_init>(tensor, sum, abs_sum_op);
}

template<int KNRows, int num_pages>
struct AttentionSum {
    using TensorT = decltype(make_tensor<float>(Shape<Int<KNRows>>{}));
    TensorT row_sum_aw;
    
    using TensorT_page = decltype(make_tensor<float>(Shape<Int<kNRows>, Int<num_pages>>{}));
    TensorT_page row_sum_aw_page;
    
    __forceinline__ __device__ AttentionSum() {
        clear(row_sum_aw_page);
    };
    // 这里从后往前遍历的块
    template<int n_block_min, int n_block_max, int page_block_size>
    __forceinline__ __device__ void update_sum_aw(Tensor0 &acc_s, int n_block){
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()))

        int Is_min_block = n_block == n_block_min;
        int Is_first_block = n_block % page_block_size == 0;
        int Is_last_block = (n_block % page_block_size == page_block_size - 1) || (n_block == n_block_max-1);
        int page_idx = n_block / page_block_size;

        if (Is_min_block){
            flash::template reduce_sum_aw</*zero_init=*/false>(scores, row_sum_aw);
            flash::template quad_allreduce_(row_sum_aw, row_sum_aw, flash::AbsSumOp<float>());
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi){row_sum_aw_page(mi, page_idx) += row_sum_aw(mi);}
        }    
        else if (Is_first_block && !Is_last_block){
            flash::template reduce_sum_aw</*zero_init=*/false>(scores, row_sum_aw);
            flash::template quad_allreduce_(row_sum_aw, row_sum_aw, flash::AbsSumOp<float>());
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi){row_sum_aw_page(mi, page_idx) += row_sum_aw(mi);}
        }
        else if (Is_first_block && Is_last_block){   // page里只有第一个block的情况
            clear(row_sum_aw);   // 从每个page的最后一个块开始遍历，需要清零
            flash::template reduce_sum_aw</*zero_init=*/true>(scores, row_sum_aw);
            flash::template quad_allreduce_(row_sum_aw, row_sum_aw, flash::AbsSumOp<float>());
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi){row_sum_aw_page(mi, page_idx) += row_sum_aw(mi);}
        }
        else if (!Is_first_block && Is_last_block){
            clear(row_sum_aw);   // 从每个page的最后一个块开始遍历，需要清零
            flash::template reduce_sum_aw</*zero_init=*/true>(scores, row_sum_aw);
        }
        else {
            flash::template reduce_sum_aw</*zero_init=*/false>(scores, row_sum_aw);
        }

    };

    __forceinline__ __device__ TensorT get_attention_weights_sum() const {
        return row_sum_aw_page;
    }
};

}  // namespace flash