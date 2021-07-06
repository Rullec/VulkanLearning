from .param_net import ParamNet
import sys

sys.path.append("../data_loader/")
from data_loader.img_data_mani import ImageDataManipulator
from data_loader.dali_data_mani import DALIDataManipulator
from net_core import cnn_net
import torch
import time
from tqdm import tqdm
import numpy as np


class CNNParamNet(ParamNet):
    '''
    Resnet Neural Network 
    receiving image input
    '''
    NAME = "CNNParamNet"
    IMAGE_DATALOADER_TYPE_KEY = "image_dataloader_type"
    INPUT_NORMALZE_MODE_KEY = "input_normalize_mode"

    def __init__(self, config_path, device):
        super().__init__(config_path, device)

    def _load_param(self):
        super()._load_param()
        self.image_dataloader_type = self.conf[
            CNNParamNet.IMAGE_DATALOADER_TYPE_KEY]

    def _build_dataloader(self):
        data_mani = ImageDataManipulator(self.conf[self.DATA_LOADER_KEY])
        # data_mani = DALIDataManipulator(self.conf[self.DATA_LOADER_KEY])
        self.train_dataloader, self.test_dataloader = data_mani.get_dataloader(
        )
        self.input_size = self.train_dataloader.get_input_size()
        self.output_size = self.train_dataloader.get_output_size()[0]

    def _build_net(self):

        # print(self.output_size)
        # exit(0)
        self.net = cnn_net(self.layers, self.output_size,
                           self.dropout).to(self.device)
        self.criterion = torch.nn.MSELoss()
        total = 0
        for i in self.net.parameters():
            total += i.numel()
        print(f"[debug] build resnet succ, total param {total}")
        # exit()

    def train(self, max_epochs=1000):
        st_time = time.time()
        # print("[debug] begin training epoch")
        for epoch in range(max_epochs):
            epoch_st_time = time.time()
            # have an iteration
            cur_epoch_train_loss = 0
            iters = 0
            total_num = 0

            total_train_cost_time = 0
            total_datafetch_cost_time = 0
            dataload_start = time.time()
            st_epoch = time.time()
            for i_batch, sampled_batched in enumerate(
                    tqdm(self.train_dataloader,
                         total=len(self.train_dataloader))):
                # profiling
                dataload_finish = time.time()
                total_datafetch_cost_time += dataload_finish - dataload_start
                train_start = time.time()
                # begin to train
                self.net.train()
                inputs, outputs = sampled_batched
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                num = inputs.shape[0]
                if num == 1:
                    continue
                self.optimizer.zero_grad()
                pred = self.net(inputs)
                loss = self.criterion(pred, outputs).to(self.device)
                if np.isnan(loss.cpu().detach()) == True:

                    print(f"input has Nan: {np.isnan(inputs.cpu()).any()}")
                    print(
                        f"pred has Nan: {np.isnan(pred.cpu().detach()).any()}")
                    print(f"gt has Nan: {np.isnan(outputs.cpu()).any()}")
                    print(f"loss: {loss.cpu()}")
                    for i in self.net.parameters():
                        print(
                            f"weight has Nan: {np.isnan( i.cpu().detach()).any()}"
                        )
                    exit(0)
                loss.backward()
                self.optimizer.step()
                train_end = time.time()
                total_train_cost_time += train_end - train_start

                # print(f"batch train cost {ed_epoch - st_epoch} s")
                # print(f"[train] single mse {loss} num {inputs.shape[0]}")
                cur_epoch_train_loss += loss.item() * num
                iters += 1
                total_num += num
                dataload_start = time.time()
                # st6 = time.time()
                # print(f"4 {st6 - st5}")
            ed_epoch = time.time()
            print(
                f"epoch cost {ed_epoch - st_epoch}, train cost {total_train_cost_time}, dataload cost {total_datafetch_cost_time}"
            )
            mean_train_loss = cur_epoch_train_loss / total_num
            # print(f"[train] total err {mean_train_loss} num {total_num}")
            # logging
            if epoch % self.iters_logging == 0:
                step = epoch / self.iters_logging
                validation_err = self._calc_validation_error()
                print(
                    f"iter {epoch} train loss {mean_train_loss:5.5f} validation loss {validation_err:5.5f}, cost {(time.time() - epoch_st_time):2.1f}, device {self.device}"
                )
                self.writer.add_scalar("train_loss", mean_train_loss, step)
                self.writer.add_scalar("validation_error", validation_err,
                                       step)
                self.writer.add_scalar("lr", self._get_lr(), step)
                # if validation_err < self.covg_threshold:
                #     break

            # saving model
            if epoch % self.iters_save_model == 0:
                name = self._get_model_save_name(float(validation_err))
                self.save_model(name)
                # print(f"name {name}")

            # update hyper parameters
            self._set_lr(max(self.lr_decay * self._get_lr(), self.min_lr))
        return validation_err