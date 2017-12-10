#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel/group_in;
			int tn = index / sz / sz / channel/group_out;
			if (tn > tc) continue;
			if (tn == tc) {
				if (th < sz / 2) continue;
				else if (th == sz / 2) {
					if (tw < sz / 2)continue;
					else weight[index] = 0;
				}
				else
					weight[index] = 0;
			}
			else
				weight[index] = 0;
		}
	}
	template <typename Dtype>
	__global__ void conv_constrains_weight_kernel_v2(const int nthreads, Dtype* const weight,
		const int channel, const int sz, const int group_in, const int group_out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % sz;
			int th = (index / sz) % sz;
			int tc = (index / sz / sz) % channel / group_in;
			int tn = index / sz / sz / channel / group_out;
			if (tn > tc) continue;
			if (tn == tc) {
				if (th < sz / 2) continue;
				else if (th == sz / 2) {
					if (tw <= sz / 2)continue;
					else weight[index] = 0;
				}
				else
					weight[index] = 0;
			}
			else
				weight[index] = 0;
		}
	}
__global__ void sync_conv_groups() { }
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	
	if (constrains_>0){
		int n = this->blobs_[0]->num();
		int ch = this->blobs_[0]->channels();
		int sz = this->blobs_[0]->width();
		if (constrains_ == 1) {
			conv_constrains_weight_kernel<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
		}
		else {
			conv_constrains_weight_kernel_v2<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >
				(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_data(), ch, sz, group_in_, group_out_);
		}
		
		CUDA_POST_KERNEL_CHECK;
	}
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!conv_need_backward_){
		return;
	}
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
