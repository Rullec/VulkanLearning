from .param_net import ParamNet
import sys

sys.path.append("../data_loader/")
from data_loader.img_data_mani import ImageDataManipulator
from net_core import cnn_net
import torch
import time
import numpy as np


class CNNParamNet(ParamNet):
    '''
    Resnet Neural Network 
    receiving image input
    '''
    NAME = "CNNParamNet"
    IMAGE_DATALOADER_TYPE_KEY = "image_dataloader_type"

    def __init__(self, config_path, device):
        super().__init__(config_path, device)

    def _load_param(self):
        super()._load_param()
        self.image_dataloader_type = self.conf[
            CNNParamNet.IMAGE_DATALOADER_TYPE_KEY]

    def _build_dataloader(self):

        data_mani = ImageDataManipulator(self.conf[self.DATA_LOADER_KEY])
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

            # have an iteration
            cur_epoch_train_loss = 0
            iters = 0
            total_num = 0

            for i_batch, sampled_batched in enumerate(
                    self.train_dataloader):
                # st1 = time.time()
                # print(i_batch)
                self.net.train()
                inputs, outputs = sampled_batched
                # inputs = np.squeeze(np.array(inputs))
                # outputs = np.squeeze(np.array(outputs))
                inputs = np.array(inputs)
                outputs = np.array(outputs)
                # print(f"outputs = {outputs}")
                # exit(0)
                st_epoch = time.time()
                # print(f"outut shape {outputs.shape}")
                inputs = torch.from_numpy(inputs).to(self.device)
                Y = torch.from_numpy(outputs).to(self.device)
                num = inputs.shape[0]
                if num == 1:
                    # print("num is 1, continue")
                    continue
                # print(
                #     f"batch {i_batch} input shape {inputs.shape} output shape {outputs.shape}"
                # )
                # print(f"num {num}")
                # outputs.type(torch.float32)
                # print(f"output tensor {outputs}")
                self.optimizer.zero_grad()
                # print(f"input shape {inputs.shape}")
                pred = self.net(inputs)

                # print(f"pred shape {pred.shape}")
                # print(f"Y shape {Y.shape}")
                # exit()
                loss = self.criterion(pred, Y).to(self.device)
                # print(f"pred {pred}")
                # print(f"Y {Y}")
                # print(f"loss {loss}")
                if np.isnan(loss.cpu().detach()) == True:

                    print(f"input has Nan: {np.isnan(inputs.cpu()).any()}")
                    print(
                        f"pred has Nan: {np.isnan(pred.cpu().detach()).any()}")
                    print(f"gt has Nan: {np.isnan(Y.cpu()).any()}")
                    print(f"loss: {loss.cpu()}")
                    for i in self.net.parameters():
                        print(
                            f"weight has Nan: {np.isnan( i.cpu().detach()).any()}"
                        )
                    exit(0)
                # else:
                #     print(f"loss is {loss}")
                loss.backward()
                self.optimizer.step()
                ed_epoch = time.time()
                # print(f"batch train cost {ed_epoch - st_epoch} s")
                # print(f"[train] single mse {loss} num {inputs.shape[0]}")
                cur_epoch_train_loss += loss.item() * num
                iters += 1
                total_num += num
                # st6 = time.time()
                # print(f"4 {st6 - st5}")

            ed_epoch = time.time()
            # print(f"epoch cost {ed_epoch - st_epoch}")
            mean_train_loss = cur_epoch_train_loss / total_num
            # print(f"[train] total err {mean_train_loss} num {total_num}")
            # logging
            if epoch % self.iters_logging == 0:
                step = epoch / self.iters_logging
                validation_err = self._calc_validation_error()
                print(
                    f"iter {epoch} train loss {mean_train_loss:5.5f} validation loss {validation_err:5.5f}, avg cost {(time.time() - st_time)/(epoch + 1):5.5f}, device {self.device}"
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