import os
import json

from numpy.random import sample
from net_core import fc_net
from data_loader import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

now = datetime.now()
writer = SummaryWriter("../log/tensorboard_log" +
                       now.strftime("%Y%m%d-%H%M%S") + "/")


class ParamNet:
    '''
    Neural Network object to inference the simulation parameters
    '''
    LEANING_RATE_KEY = "lr"
    MIN_LEANING_RATE_KEY = "min_lr"
    LEANING_RATE_DECAY_KEY = "lr_decay"
    WEIGHT_DECAY_KEY = "weight_decay"
    CONVERGENCE_THRESHOLD_KEY = "covg_threshold"
    DATA_DIR_KEY = "data_dir"
    LAYERS_KEY = "layers"
    MODEL_OUTPUT_DIR_KEY = "model_output_dir"
    LOAD_MODEL_PATH_KEY = "model_path"
    SAVE_MODEL_ITERS_KEY = "save_model_iters"
    LOGGING_ITERS_KEY = "logging_iters"
    BATCH_SIZE_KEY = "batch_size"
    OPTIMIZER_TYPE_KEY = "optimizer"
    ENABLE_LOG_PREDICTION_KEY = "enable_log_prediction"

    def __init__(self, config_path, device):
        '''
        :param config_path: config string path
        :param device: cuda or cpu?
        '''
        self.device = device
        self.conf_path = config_path
        with open(config_path) as f:
            self.conf = json.load(f)
        self._load_param()
        self._build_dataloader()
        self._build_net()
        self._build_optimizer()
        self._postprocess()

    def _load_param(self):
        '''
        Load important parameter from the "self.conf" dict
        Note that some parameter is optional
        '''
        self.lr = float(self.conf[ParamNet.LEANING_RATE_KEY])
        self.min_lr = float(self.conf[ParamNet.MIN_LEANING_RATE_KEY])
        self.lr_decay = float(self.conf[ParamNet.LEANING_RATE_DECAY_KEY])
        self.weight_decay = float(self.conf[ParamNet.WEIGHT_DECAY_KEY])
        self.covg_threshold = float(
            self.conf[ParamNet.CONVERGENCE_THRESHOLD_KEY])
        self.data_dir = str(self.conf[ParamNet.DATA_DIR_KEY])
        self.model_output_dir = self.conf[ParamNet.MODEL_OUTPUT_DIR_KEY]
        self.batch_size = self.conf[ParamNet.BATCH_SIZE_KEY]
        # optional
        self.load_model_path = str(
            self.conf[ParamNet.LOAD_MODEL_PATH_KEY]
        ) if ParamNet.LOAD_MODEL_PATH_KEY in self.conf else None
        self.iters_save_model = int(self.conf[ParamNet.SAVE_MODEL_ITERS_KEY])
        self.iters_logging = int(self.conf[ParamNet.LOGGING_ITERS_KEY])
        self.optimizer_type = self.conf[ParamNet.OPTIMIZER_TYPE_KEY]
        self.enable_log_prediction = self.conf[ParamNet.ENABLE_LOG_PREDICTION_KEY]

    def save_model(self, name):
        '''
        Save model to the given path "name"
        '''
        tar_dir = os.path.dirname(name)
        if os.path.exists(tar_dir) == False:
            print(
                f"[warn] save model target dir {tar_dir} is empty, create it")
            os.makedirs(tar_dir)
        print(f"[log] save model to {name}")
        torch.save(self.net.state_dict(), name)

    def load_model(self, name):
        '''
        Load model from the given path "name"
        '''
        print(f"[log] load model from {name}")
        self.net.load_state_dict(torch.load(name))

    def _build_net(self):
        '''
        Build my network strucutre from given "layers"
        '''
        self.layers = list(self.conf[ParamNet.LAYERS_KEY])
        for i in self.layers:
            assert type(i) == int, f"layers = {i}"
        self.net = fc_net(self.input_size, self.layers, self.output_size,
                          self.device).to(self.device)
        self.criterion = torch.nn.MSELoss()

    def _build_dataloader(self):
        '''
        Create dataloader and get the data size
        '''
        self.data_loader = DataLoader(
            self.data_dir,
            0.8,
            0.2,
            self.batch_size,
            enable_log_prediction=self.enable_log_prediction)
        self.input_size = self.data_loader.get_input_size()
        self.output_size = self.data_loader.get_output_size()

    def _build_optimizer(self):
        if self.optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(),
                                       lr=self.lr,
                                       weight_decay=self.weight_decay)
        elif self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.net.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)

    def _postprocess(self):
        # 1. check the target loading model
        if os.path.exists(self.load_model_path) == False:
            print(
                f"[warning] given load path {self.load_model_path} is invalid, ignore"
            )
            self.load_model_path = None
        if self.load_model_path is not None:
            self.load_model(self.load_model_path)

    def _get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

    def _set_lr(self, lr):
        for g in self.optimizer.param_groups:
            if 'lr' in g:
                g['lr'] = lr

    def _get_model_save_name(self, cur_loss):
        import datetime
        import os
        output_name = datetime.datetime.now().strftime("%m-%d-%H_%M_%S")
        output_name = f"{output_name}-{str(cur_loss)[:5]}.pkl"
        output_name = os.path.join(self.model_output_dir, output_name)
        return output_name

    def train(self):

        X, Y = self.data_loader.load_data()
        X = torch.from_numpy(X).float().to(self.device)
        Y = torch.from_numpy(Y).float().to(self.device)
        iters = int(1e6)
        st_time = time.time()
        for i in range(iters):
            self.optimizer.zero_grad()
            pred = self.net(X)
            # print(pred)
            loss = self.criterion(pred, Y).to(self.device)
            loss.backward()
            self.optimizer.step()

            # logging
            if i % self.iters_logging == 0:
                print(
                    f"iter {i} loss {loss}, avg cost {(time.time() - st_time)/(i + 1)}, device {self.device}"
                )
                writer.add_scalar("loss", loss, i / 100)
                print(f"pred = {pred.cpu()[0]}")
                print(f"ground_truth = {Y.cpu()[0]}")
                if loss < self.covg_threshold:
                    break

            # saving model
            if i % self.iters_save_model == 0:
                name = self._get_model_save_name(loss.item())
                self.save_model(name)
                # print(f"name {name}")
        print("finished training")

    def _calc_validation_error(self):
        # print("-----begin validation---")
        iters = 0
        total_validation_err = 0
        total_num = 0
        for sampled_batched in self.data_loader.get_validation_data():
            inputs, outputs = sampled_batched
            inputs = np.array(inputs)
            outputs = np.array(outputs)
            num = inputs.shape[0]
            # print(f"outut shape {outputs.shape}")
            inputs = torch.from_numpy(inputs).to(self.device)
            Y = torch.from_numpy(outputs).to(self.device)
            # self.optimizer.zero_grad()
            pred = self.net(inputs)
            single_mse = self.criterion(pred, Y)
            total_validation_err += single_mse * num
            # print(f"[valid] single mse {single_mse} num {num}")
            iters += 1
            total_num += num
        output_mean = self.data_loader.get_output_mean()
        output_std = self.data_loader.get_output_std()
        # print(f"output mean {output_mean}")
        # print(f"output std {output_std}")
        np_pred = pred.cpu()[0].detach().numpy()
        np_gt = Y.cpu()[0].detach().numpy()

        # print(f"[valid] diff = {(np_pred - np_gt) * output_std}")
        valdation_err = total_validation_err / total_num
        # print(f"[valid] total err {total_validation_err} num {total_num}, val err = {valdation_err}")
        # print("-----end validation---")
        return valdation_err

    def train_generator(self):
        # self.train_loader, self.validation_loader = self.data_loader.get_torch_dataloader(
        # )
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
                # print(
                #     f"batch {i_batch} input shape {inputs.shape} output shape {outputs.shape}"
                # )

                # outputs.type(torch.float32)
                # print(f"output tensor {outputs}")
                self.optimizer.zero_grad()
                pred = self.net(inputs)
                loss = self.criterion(pred, Y).to(self.device)
                loss.backward()
                self.optimizer.step()
                # print(f"[train] single mse {loss} num {inputs.shape[0]}")
                cur_epoch_train_loss += loss * num
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
                writer.add_scalar("train_loss", mean_train_loss, step)
                writer.add_scalar("validation_error",
                                  self._calc_validation_error(), step)
                writer.add_scalar("lr", self._get_lr(), step)
                # if validation_err < self.covg_threshold:
                #     break

            # saving model
            if epoch % self.iters_save_model == 0:
                name = self._get_model_save_name(loss.item())
                self.save_model(name)
                # print(f"name {name}")

            # update hyper parameters
            self._set_lr(max(self.lr_decay * self._get_lr(), self.min_lr))

    def _calc_validation_error_percentage(self, prediction, gt, output_mean,
                                          output_std):
        # print(f"raw pred {prediction}")
        # print(f"output std {output_std}")
        # print(f"res {prediction * output_std}")
        pred = prediction * output_std + output_mean
        gt = gt * output_std + output_mean
        exp_pred = np.exp(pred)
        exp_gt = np.exp(gt)
        exp_diff = np.abs(exp_pred - exp_gt)
        diff_perc = np.array(exp_diff / exp_gt * 100)
        diff_perc = diff_perc.reshape(-1, )
        # print(f"diff perc {diff_perc}")
        # exit(0)
        return list(diff_perc)

    def test(self):
        iters = 0
        total_validation_err = 0
        total_num = 0
        diff_perc_lst = []
        output_mean = self.data_loader.get_output_mean()
        output_std = self.data_loader.get_output_std()
        for i_batch, sampled_batched in enumerate(
                self.data_loader.get_validation_data()):
            # st1 = time.time()
            # print(i_batch)
            inputs, outputs = sampled_batched
            inputs = np.array(inputs)
            outputs = np.array(outputs)
            # print(f"outut shape {outputs.shape}")
            inputs = torch.from_numpy(inputs).to(self.device)
            Y = torch.from_numpy(outputs).to(self.device)
            num = inputs.shape[0]
            # self.optimizer.zero_grad()
            pred = self.net(inputs)
            single_mse = self.criterion(pred, Y)
            total_validation_err += single_mse * num
            diff_perc_lst += self._calc_validation_error_percentage(
                Y.cpu().detach(),
                pred.cpu().detach(), output_mean, output_std)
            # print(f"[valid] single mse {single_mse} num {num}")
            # print(diff_perc_lst)
            # exit(0)
            iters += 1
            total_num += num

        # print(f"output mean {output_mean}")
        # print(f"output std {output_std}")
        np_pred = pred.cpu().detach().numpy()
        np_gt = Y.cpu().detach().numpy()
        diff = np_pred - np_gt
        print_samples = 100
        idx = np.random.permutation(np_pred.shape[0])[:print_samples]
        assert self.enable_log_prediction == True

        for i in idx:
            # print(f"pred {np_pred[i, :]}, gt {np_gt[i, :]}, diff {diff[i, :]}")
            pred = np_pred[i, :] * output_std + output_mean
            gt = np_gt[i, :] * output_std + output_mean
            # print(f"raw gt = {np_gt[i, :]}")
            # print(f"output std = {output_std}")
            # print(f"output mean = {output_mean}")
            # print(f"later gt = {gt}")
            exp_pred = np.exp(pred)
            exp_gt = np.exp(gt)
            exp_diff = np.abs(exp_pred - exp_gt)
            diff_perc = exp_diff / exp_gt * 100
            print(
                f"pred {exp_pred}\ngt {exp_gt}\ndiff {exp_diff}\nperc {diff_perc}\n"
            )
            for i in list(diff_perc):
                diff_perc_lst.append(i)
            # print(f"diff {diff[i, :]}")
        import matplotlib.pyplot as plt
        sorted_lst = sorted(diff_perc_lst)
        print(f"50% {sorted_lst[int(0.5 * len(sorted_lst))]}")
        print(f"80% {sorted_lst[int(0.8 * len(sorted_lst))]}")
        print(f"90% {sorted_lst[int(0.9 * len(sorted_lst))]}")
        print(f"99% {sorted_lst[int(0.99 * len(sorted_lst))]}")
        print(f"99.9% {sorted_lst[int(0.999 * len(sorted_lst))]}")
        print(sorted_lst)
        plt.hist(diff_perc_lst)
        # print(diff_perc_lst)
        plt.show()

        # print(f"output std = {output_std}")