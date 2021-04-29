from param_net import ParamNet
from image_data_loader import ImageDataLoader
from net_core import res_net
import torch
import time
import numpy as np


class ResParamNet(ParamNet):
    '''
    Resnet Neural Network 
    receiving image input
    '''
    NAME = "ResParamNet"

    def __init__(self, config_path, device):
        super().__init__(config_path, device)
        # print("build res ParamNet")

    def _build_dataloader(self):
        print("[log] begin to build dataloader in resnet")
        self.data_loader = ImageDataLoader(
            self.data_dir,
            0.8,
            0.2,
            self.batch_size,
            enable_log_prediction=self.enable_log_prediction)
        self.input_size = self.data_loader.get_input_size()
        self.output_size = self.data_loader.get_output_size()[0]
        # print(f"input size {self.input_size}")
        # print(f"output size {self.output_size}")
        # exit(0)

    def _build_net(self):
        layers = [16]
        # print(self.output_size)
        # exit(0)
        self.net = res_net(layers, self.output_size).to(self.device)
        self.criterion = torch.nn.MSELoss()
        total = 0
        for i in self.net.parameters():
            total += i.numel()
        print(f"[debug] build resnet succ, total param {total}")
        # exit()

    def train(self):
        max_epochs = 10000
        st_time = time.time()
        print("[debug] begin training epoch")
        for epoch in range(max_epochs):

            # have an iteration
            cur_epoch_train_loss = 0
            iters = 0
            total_num = 0
            st_epoch = time.time()
            self.data_loader.shuffle()
            for i_batch, sampled_batched in enumerate(
                    self.data_loader.get_train_data()):
                # st1 = time.time()
                # print(i_batch)
                inputs, outputs = sampled_batched
                inputs = np.array(inputs)
                outputs = np.array(outputs)
                # print(f"outputs = {outputs}")
                # exit(0)
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

                # print(f"pred type {pred.dtype}")
                # print(f"Y type {Y.dtype}")
                loss = self.criterion(pred, Y).to(self.device)
                # print(f"pred {pred}")
                # print(f"Y {Y}")
                # print(f"loss {loss}")
                # exit(0)
                loss.backward()
                self.optimizer.step()
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
                    f"iter {epoch} train loss {mean_train_loss} validation loss {validation_err}, avg cost {(time.time() - st_time)/(epoch + 1)}, device {self.device}"
                )
                self.writer.add_scalar("train_loss", mean_train_loss, step)
                self.writer.add_scalar("validation_error",
                                       self._calc_validation_error(), step)
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