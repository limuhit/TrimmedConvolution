#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "multobin_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void MulToBinLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		MulToBinParameter mp = this->layer_param_.multobin_param();
		levels_ = mp.levels();

	}
	template <typename Dtype>
	void MulToBinLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		width_ = bottom[0]->width();
		height_ = bottom[0]->height();
		channel_ = bottom[0]->channels();
		num_ = bottom[0]->num();
		inner_size_ = width_*height_;
		top[0]->Reshape(num_,channel_*levels_,height_,width_);
	}
	template <typename Dtype>
	void multobin_forward_cpu_kernel(const int num, const Dtype * const bottom, Dtype * const top,
		const int inner_size, const int channel, const int level) {
		for (int i = 0; i < num; i++) {
			int ts = i % inner_size;
			int tc = (i / inner_size) % channel;
			int tn = (i / inner_size / channel);
			int pbase = tn*channel*level*inner_size + tc*level*inner_size+ ts;
			unsigned int data = static_cast <int>(bottom[i]);
			for (int j = 0; j < level; j++)
			{
				top[pbase] = data % 2;
				data = data >> 1;
				pbase += inner_size;
			}
		}
	}
	template <typename Dtype>
	void MulToBinLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		
		multobin_forward_cpu_kernel<Dtype>(bottom[0]->count(), bottom_data, top[0]->mutable_cpu_data(), inner_size_, channel_, levels_);

	}

	template <typename Dtype>
	void MulToBinLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		Dtype* bottom_dif = bottom[0]->mutable_cpu_diff();


	}

#ifdef CPU_ONLY
	STUB_GPU(MulToBinLayer);
#endif

	INSTANTIATE_CLASS(MulToBinLayer);
	REGISTER_LAYER_CLASS(MulToBin);

}  // namespace caffe
