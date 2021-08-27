from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from .dali_pipe import DALIDataAugPipeline


def reshape_input(raw_img, num_of_views):
    num_of_imgs = raw_img.shape[0]
    num_of_data = int(num_of_imgs / num_of_views)
    assert num_of_imgs % num_of_views == 0
    raw_img = raw_img.reshape(num_of_data, num_of_views, raw_img.shape[1],
                              raw_img.shape[2])
    # raw_img = torch.Tensor(torch.split(raw_img, num_of_data))
    # exit()
    return raw_img


class DALITorchWrapper:
    def __init__(self, file_reader_for_shuffle, dali_generic_iterator,
                 input_key, output_key, num_of_view, input_mean, input_std,
                 output_mean, output_std):
        self.dali_generic_iterator = dali_generic_iterator
        self.file_reader_for_shuffle = file_reader_for_shuffle
        self.input_key = input_key
        self.output_key = output_key
        self.num_of_view = num_of_view
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std

    def __iter__(self):
        return self

    def __next__(self):
        try:
            cont_dict = next(self.dali_generic_iterator)[0]
            input = cont_dict[self.input_key]
            input = reshape_input(input, self.num_of_view)
            label = cont_dict[self.output_key]
            label = label[::self.num_of_view, :]
            return input, label
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.file_reader_for_shuffle)

    def reset(self):
        self.file_reader_for_shuffle.shuffle()
        self.dali_generic_iterator.reset()
        # print(f"reset")

    def get_input_size(self):
        return self.input_mean.shape

    def get_output_size(self):
        return self.output_mean.shape

import os
def build_dali_torch_wrapper(file_reader, batch_size):
    
    train_pipe = DALIDataAugPipeline(batch_size=batch_size,
                                     num_threads=os.cpu_count(),
                                     device_id=0,
                                     external_data=file_reader)

    train_pipe.build()

    train_iter = DALIGenericIterator(train_pipe, ['data', 'label'],
                                     last_batch_padded=True,
                                     last_batch_policy=LastBatchPolicy.PARTIAL,
                                     dynamic_shape=True)

    train_iter = DALITorchWrapper(file_reader, train_iter, 'data', 'label',
                                  file_reader.num_of_view,
                                  file_reader.input_mean,
                                  file_reader.input_std,
                                  file_reader.output_mean,
                                  file_reader.output_std)
    return train_iter