import os
import json
from net_core import fc_net
from data_loader import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../log/tensorboard_log")


class ParamNet:
    LEANING_RATE_KEY = "lr"
    LEANING_RATE_DECAY_KEY = "lr_decay"
    WEIGHT_DECAY_KEY = "weight_decay"
    CONVERGENCE_THRESHOLD_KEY = "covg_threshold"
    DATA_DIR_KEY = "data_dir"
    LAYERS_KEY = "layers"
    MODEL_OUTPUT_DIR_KEY = "model_output_dir"
    LOAD_MODEL_PATH_KEY = "model_path"
    SAVE_MODEL_ITERS_KEY = "save_model_iters"
    LOGGING_ITERS_KEY = "logging_iters"

    def __init__(self, config_path, device):
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
        self.lr_decay = float(self.conf[ParamNet.LEANING_RATE_DECAY_KEY])
        self.weight_decay = float(self.conf[ParamNet.WEIGHT_DECAY_KEY])
        self.covg_threshold = float(
            self.conf[ParamNet.CONVERGENCE_THRESHOLD_KEY])
        self.data_dir = str(self.conf[ParamNet.DATA_DIR_KEY])
        self.model_output_dir = self.conf[ParamNet.MODEL_OUTPUT_DIR_KEY]

        # optional
        self.load_model_path = str(
            self.conf[ParamNet.LOAD_MODEL_PATH_KEY]
        ) if ParamNet.LOAD_MODEL_PATH_KEY in self.conf else None
        self.iters_save_model = int(self.conf[ParamNet.SAVE_MODEL_ITERS_KEY])
        self.iters_logging = int(self.conf[ParamNet.LOGGING_ITERS_KEY])

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
        self.data_loader = DataLoader(self.data_dir)
        self.input_size = self.data_loader.get_input_size()
        self.output_size = self.data_loader.get_output_size()

    def _build_optimizer(self):
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
        log_Y = np.log(Y)
        log_Y = torch.from_numpy(log_Y).float().to(self.device)
        iters = int(1e6)
        st_time = time.time()
        for i in range(iters):
            self.optimizer.zero_grad()
            pred = self.net(X)
            # print(pred)
            loss = self.criterion(pred, log_Y).to(self.device)
            loss.backward()
            self.optimizer.step()

            # logging
            if i % self.iters_logging == 0:
                print(
                    f"iter {i} loss {loss}, avg cost {(time.time() - st_time)/(i + 1)}, device {self.device}"
                )
                writer.add_scalar("loss", loss, i / 100)
                print(f"pred = {pred.cpu()[0]}")
                print(f"ground_truth = {log_Y.cpu()[0]}")
                if loss < self.covg_threshold:
                    break

            # saving model
            if i % self.iters_save_model == 0:
                name = self._get_model_save_name(loss.item())
                self.save_model(name)
                # print(f"name {name}")
        print("finished training")