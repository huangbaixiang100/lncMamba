from nnLayer import *
from metrics import *
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.distributed
import torch,time,os
import numpy as np



class BaseClassifier:
    def __init__(self, model):
        pass

    def calculate_y_logit(self, X):
        pass

    def calculate_y_prob(self, X):
        pass

    def calculate_y(self, X):
        pass

    def calculate_y_prob_by_iterator(self, dataStream):
        pass

    def calculate_y_by_iterator(self, dataStream):
        pass

    def calculate_loss(self, X, Y):
        pass

    def train(self, optimizer, trainDataSet, validDataSet=None, otherDataSet=None,
              batchSize=256, epoch=100, earlyStop=10, saveRounds=1,
              isHigherBetter=False, metrics="LOSS", report=["LOSS"],
              attackTrain=False, attackLayerName='emb', useEMA=False, prefetchFactor=2,
              savePath='model', shuffle=True, dataLoadNumWorkers=0, pinMemory=False,
              trainSampler=None, validSampler=None, warmupEpochs=0, doEvalTrain=True, doEvalValid=True,
              doEvalOther=False):
        if attackTrain:
            self.fgm = FGM(self.model, emb_name=attackLayerName)
        if useEMA:
            ema = EMA(self.model, 0.999)
            ema.register()

        metrictor = self.metrictor if hasattr(self, "metrictor") else Metrictor()
        device = next(self.model.parameters()).device
        worldSize = torch.distributed.get_world_size() if self.mode > 0 else 1
        # schedulerRLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if isHigherBetter else 'min', factor=0.5, patience=20, verbose=True)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        itersPerEpoch = (len(trainDataSet) + batchSize - 1) // batchSize
        warmSteps = int(itersPerEpoch * warmupEpochs / worldSize)
        decaySteps = int(itersPerEpoch * epoch / worldSize) - warmSteps
        schedulerRLR = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lambda i: i / warmSteps if i < warmSteps else (
                                                                                                                             decaySteps - (
                                                                                                                                 i - warmSteps)) / decaySteps)
        trainStream = DataLoader(trainDataSet, batch_size=batchSize, shuffle=shuffle, num_workers=dataLoadNumWorkers,
                                 pin_memory=pinMemory, collate_fn=self.collateFunc, sampler=trainSampler,
                                 prefetch_factor=prefetchFactor)
        evalTrainStream = DataLoader(trainDataSet, batch_size=batchSize, shuffle=False, num_workers=dataLoadNumWorkers,
                                     pin_memory=pinMemory, collate_fn=self.collateFunc, sampler=trainSampler,
                                     prefetch_factor=prefetchFactor)

        mtc, bestMtc, stopSteps = 0.0, 0.0 if isHigherBetter else 9999999999, 0
        if validDataSet is not None: validStream = DataLoader(validDataSet, batch_size=batchSize, shuffle=False,
                                                              num_workers=dataLoadNumWorkers, pin_memory=pinMemory,
                                                              collate_fn=self.collateFunc, sampler=validSampler,
                                                              prefetch_factor=prefetchFactor)
        if otherDataSet is not None: otherStream = DataLoader(otherDataSet, batch_size=batchSize, shuffle=False,
                                                              num_workers=dataLoadNumWorkers, pin_memory=pinMemory,
                                                              collate_fn=self.collateFunc, sampler=validSampler,
                                                              prefetch_factor=prefetchFactor)
        st = time.time()
        for e in range(epoch):
            pbar = tqdm(trainStream)
            self.to_train_mode()
            for data in pbar:
                data = dict_to_device(data, device=device)
                loss = self._train_step(data, optimizer, attackTrain)
                if useEMA:
                    ema.update()
                schedulerRLR.step()
                pbar.set_description(
                    f"Training Loss: {loss}; Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
            if useEMA:
                ema.apply_shadow()
            if ((self.mode > 0 and torch.distributed.get_rank() == 0) or self.mode == 0):
                if (validDataSet is not None) and ((e + 1) % saveRounds == 0):
                    print(f'========== Epoch:{e + 1:5d} ==========')
                    with torch.no_grad():
                        self.to_eval_mode()
                        if doEvalTrain:
                            print(f'[Total Train]', end='')
                            # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                            # metrictor.show_res(res)
                            data = self.calculate_y_prob_by_iterator(evalTrainStream)
                            metrictor.set_data(data)
                            # print()
                            metrictor(report)
                        if doEvalValid:
                            print(f'[Total Valid]', end='')
                            # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                            # metrictor.show_res(res)
                            data = self.calculate_y_prob_by_iterator(validStream)
                            metrictor.set_data(data)
                            res = metrictor(report)
                            mtc = res[metrics]
                            print('=================================')
                            if (mtc > bestMtc and isHigherBetter) or (mtc < bestMtc and not isHigherBetter):
                                if (self.mode > 0 and torch.distributed.get_rank() == 0) or self.mode == 0:
                                    print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                                    bestMtc = mtc
                                    self.save("%s.pkl" % savePath, e + 1, bestMtc)

                                stopSteps = 0
                            else:
                                stopSteps += 1
                                if stopSteps >= earlyStop:
                                    print(
                                        f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e + 1}, stop training.')
                                    break
                        if doEvalOther:
                            print(f'[Total Other]', end='')
                            # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                            # metrictor.show_res(res)
                            data = self.calculate_y_prob_by_iterator(otherStream)
                            metrictor.set_data(data)
                            # print()
                            metrictor(report)
            if useEMA:
                ema.restore()
        if (self.mode > 0 and torch.distributed.get_rank() == 0) or self.mode == 0:
            with torch.no_grad():
                new_path = "%s_%s.pkl" % (savePath, ("%.3lf" % bestMtc)[2:])
               # os.rename("%s.pkl" % savePath, new_path)
               # self.load(new_path)  # 使用新路径加载
                self.load(savePath)
                self.to_eval_mode()
            #    os.rename("%s.pkl" % savePath, "%s_%s.pkl" % (savePath, ("%.3lf" % bestMtc)[2:]))
                print(f'============ Result ============')
                print(f'[Total Train]', end='')
                # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                # metrictor.show_res(res)
                data = self.calculate_y_prob_by_iterator(evalTrainStream)
                metrictor.set_data(data)
                metrictor(report)
                print(f'[Total Valid]', end='')
                # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                # metrictor.show_res(res)
                data = self.calculate_y_prob_by_iterator(validStream)
                metrictor.set_data(data)
                res = metrictor(report)
                if otherDataSet is not None:
                    print(f'[Total Other]', end='')
                    # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                    # metrictor.show_res(res)
                    data = self.calculate_y_prob_by_iterator(otherStream)
                    metrictor.set_data(data)
                    metrictor(report)
                # metrictor.each_class_indictor_show(dataClass.id2lab)
                print(f'================================')
                return res

    def to_train_mode(self):
        self.model.train()  # set the module in training mode
        if self.collateFunc is not None:
            self.collateFunc.train = True

    def to_eval_mode(self):
        self.model.eval()
        if self.collateFunc is not None:
            self.collateFunc.train = False

    def _train_step(self, data, optimizer, attackTrain):
        loss = self.calculate_loss(data)
        loss.backward()
        if attackTrain:
            self.fgm.attack()
            lossAdv = self.calculate_loss(data)
            lossAdv.backward()
            self.fgm.restore()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        optimizer.zero_grad()
        return loss

    def save(self, path, epochs, bestMtc=None):
        # 确保路径正确，如果没有以 .pkl 结尾，则添加 .pkl 后缀
        if not path.endswith('.pkl'):
            path = path + '.pkl'  # 确保路径是以 .pkl 结尾
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc, 'model': self.model.state_dict()}
        torch.save(stateDict, path)
        print(f'Model saved in "{path}".')

    def load(self, path, map_location=None):
        # 获取文件夹中的所有 .pkl 文件
        pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]

        # 如果没有找到 .pkl 文件，抛出异常
        if not pkl_files:
            raise FileNotFoundError("No .pkl files found in the provided directory.")

        # 获取文件路径以及对应的修改时间
        pkl_files_with_time = [(f, os.path.getmtime(os.path.join(path, f))) for f in pkl_files]

        # 找到修改时间最新的文件
        latest_file = max(pkl_files_with_time, key=lambda x: x[1])[0]

        # 构造最新文件的完整路径
        latest_file_path = os.path.join(path, latest_file)

        # 加载最新权重
        parameters = torch.load(latest_file_path, map_location=map_location)
        self.model.load_state_dict(parameters['model'])

        print("%d epochs and %.3lf val Score 's model load finished." % (parameters['epochs'], parameters['bestMtc']))

class PadAndTknizeCollateFunc:
    def __init__(self, tokenizer, maskProb=0.15, groups=-1, duplicate=False, randomSample=False, dataEnhance=False,
                 dataEnhanceRatio=0.1):
        self.tokenizer = tokenizer
        self.maskProb = maskProb
        self.train = False
        self.duplicate = duplicate
        self.randomSample = randomSample
        self.dataEnhance = dataEnhance
        self.dataEnhanceRatio = dataEnhanceRatio
        self.groups = groups

    def __call__(self, data):
        tmp = [i['sequence'] for i in data]

        if data[0]['tokenizedKgpSeqArr'] is not None:
            tokenizedKgpSeqArr = torch.cat([i['tokenizedKgpSeqArr'].unsqueeze(0) for i in data], dim=0)
            tokenizedSeqArr, maskPAD, posIdxArr = None, None, None
        elif self.groups > -1:
            if self.randomSample and self.train and random.random() < 0.2:  # 随机丢失15的核苷酸
                tmp = [[j for j in i if random.random() > 0.15] for i in tmp]
            tokenizedKgpSeqArr = torch.tensor(self.tokenizer.tokenize_sentences_to_k_group(tmp, self.groups),
                                              dtype=torch.float32)
            tokenizedSeqArr, maskPAD, posIdxArr = None, None, None
        else:
            tokenizedKgpSeqArr = None

            if self.randomSample and self.train:
                #             posIdxArr = [np.sort(np.random.permutation(len(i))[:self.tokenizer.seqMaxLen]) for i in tmp]
                posIdxArr = [
                    [np.int(random.random() * (len(i) - self.tokenizer.seqMaxLen)), self.tokenizer.seqMaxLen] if len(
                        i) > self.tokenizer.seqMaxLen else [0, len(i)] for i in tmp]
                posIdxArr = [np.arange(i, i + j) for i, j in posIdxArr]
            else:
                posIdxArr = None
            tokenizedSeqArr, maskPAD = self.tokenizer.tokenize_sentences(
                [np.array(i)[posIdx] for i, posIdx in zip(tmp, posIdxArr)] if (
                            self.randomSample and posIdxArr is not None) else tmp,
                train=self.train)  # batchSize × seqLen
            if posIdxArr is not None:
                seqMaxLen = min(max([len(i) for i in tmp]),
                                self.tokenizer.seqMaxLen) if self.train else self.tokenizer.seqMaxLen
                posIdxArr = [[0] + (i + 1).tolist() + (
                    list(range(j['sLen'] + 1, seqMaxLen + 1)) if (len(tmp) > 1 or self.train) else []) for i, j in
                             zip(posIdxArr, data)]
                posIdxArr = torch.tensor(posIdxArr, dtype=torch.float32)

            tokenizedSeqArr, maskPAD = torch.tensor(tokenizedSeqArr, dtype=torch.long), torch.tensor(maskPAD,
                                                                                                     dtype=torch.bool)
            if self.duplicate:
                tokenizedSeqArr = torch.cat([tokenizedSeqArr, tokenizedSeqArr], dim=0)
                maskPAD = torch.cat([maskPAD, maskPAD], dim=0)
                posIdxArr = torch.cat([posIdxArr, posIdxArr], dim=0)
            maskPAD = maskPAD.reshape(len(tokenizedSeqArr), 1, -1) & maskPAD.reshape(len(tokenizedSeqArr), -1, 1)

            seqLens = torch.tensor([min(i['sLen'], self.tokenizer.seqMaxLen) for i in data], dtype=torch.int32)
            if self.dataEnhance:
                for i in range(len(tokenizedSeqArr)):  # 数据增强
                    if random.random() < self.dataEnhanceRatio / 2:  # 随机排列
                        tokenizedSeqArr[i][:seqLens[i]] = tokenizedSeqArr[i][:seqLens[i]][
                            np.random.permutation(int(seqLens[i]))]
                    if random.random() < self.dataEnhanceRatio:  # 逆置
                        tokenizedSeqArr[i][:seqLens[i]] = tokenizedSeqArr[i][:seqLens[i]][range(int(seqLens[i]))[::-1]]

        tmp = self.tokenizer.tokenize_labels([i['label'] for i in data])
        labArr = np.zeros((len(tmp), self.tokenizer.labNum))
        for i in range(len(tmp)):
            labArr[i, tmp[i]] = 1
        tokenizedLabArr = torch.tensor(labArr, dtype=torch.float32)  # batchSize
        if self.duplicate:
            tokenizedLabArr = torch.cat([tokenizedLabArr, tokenizedLabArr], dim=0)

        if self.tokenizer.useAAC:
            aacFea = torch.tensor(self.tokenizer.transform_to_AAC([i['sequence_'] for i in data]), dtype=torch.float32)
        else:
            aacFea = None

        return {'tokenizedSeqArr': tokenizedSeqArr, 'tokenizedKgpSeqArr': tokenizedKgpSeqArr, 'maskPAD': maskPAD,
                'posIdxArr': posIdxArr,
                'tokenizedLabArr': tokenizedLabArr, 'aacFea': aacFea}  #


class SequenceClassifier(BaseClassifier):
    def __init__(self, model, collateFunc=None, mode=0, criterion=None):
        self.model = model
        self.collateFunc = collateFunc
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.mode = mode
        if mode == 2:
            self.scaler = torch.cuda.amp.GradScaler()
        elif mode == 3:
            import apex

    def calculate_y_logit(self, data):
        return self.model(data)

    def calculate_y_prob(self, data):
        Y_pre = self.calculate_y_logit(data)['y_logit']
        return {'y_prob': F.softmax(Y_pre, dim=-1)}

    def calculate_y(self, data):
        Y_pre = self.calculate_y_prob(data)['y_prob']
        return {'y_pre': (Y_pre > 0.5).astype('int32')}

    def calculate_loss_by_iterator(self, dataStream):
        loss, cnt = 0, 0
        for data in dataStream:
            loss += self.calculate_loss(data) * len(data['tokenizedSeqArr'])
            cnt += len(data['tokenizedSeqArr'])
        return loss / cnt

    def calculate_y_prob_by_iterator(self, dataStream):
        device = next(self.model.parameters()).device
        YArr, Y_preArr = [], []
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            Y_pre, Y = self.calculate_y_prob(data)['y_prob'].detach().cpu().data.numpy().astype('float32'), data[
                'tokenizedLabArr'].detach().cpu().data.numpy().astype('int32')
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr, Y_preArr = np.vstack(YArr).astype('int32'), np.vstack(Y_preArr).astype('float32')
        return {'y_prob': Y_preArr, 'y_true': YArr}

    def calculate_loss(self, data):
        out = self.calculate_y_logit(data)
        Y = data['tokenizedLabArr']
        Y_logit = out['y_logit'].reshape(len(Y), -1)
        return self.criterion(Y_logit, Y)

    def _train_step(self, data, optimizer, attackTrain):
        optimizer.zero_grad()
        loss = self.calculate_loss(data)
        loss.backward()
        if attackTrain:
            self.fgm.attack()
            lossAdv = self.calculate_loss(data)
            lossAdv.backward()
            self.fgm.restore()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        return loss



    def save(self, path, epochs, bestMtc=None):
        if self.mode == 0:
            model = self.model.state_dict()
        else:
            model = self.model.module.state_dict()
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc, 'model': model}
        torch.save(stateDict, path)
        print('Model saved in "%s".' % path)

    def load(self, path, map_location=None):
        # 获取文件夹中的所有 .pkl 文件
        pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]

        # 如果没有找到 .pkl 文件，抛出异常
        if not pkl_files:
            raise FileNotFoundError("No .pkl files found in the provided directory.")

        # 获取文件路径以及对应的修改时间
        pkl_files_with_time = [(f, os.path.getmtime(os.path.join(path, f))) for f in pkl_files]

        # 找到修改时间最新的文件
        latest_file = max(pkl_files_with_time, key=lambda x: x[1])[0]

        # 构造最新文件的完整路径
        latest_file_path = os.path.join(path, latest_file)

        # 加载最新权重
        parameters = torch.load(latest_file_path, map_location=map_location)
        self.model.load_state_dict(parameters['model'])

        print("%d epochs and %.3lf val Score 's model load finished." % (parameters['epochs'], parameters['bestMtc']))

class SequenceMultiLabelClassifier(SequenceClassifier):
    def __init__(self, model, collateFunc=None, mode=0, criterion=None):
        self.model = model
        self.collateFunc = collateFunc
        self.criterion = nn.MultiLabelSoftMarginLoss() if criterion is None else criterion
        self.mode = mode
        if mode == 2:
            self.scaler = torch.cuda.amp.GradScaler()
        elif mode == 3:
            import apex

    def calculate_y_prob(self, data):
        Y_pre = self.calculate_y_logit(data)['y_logit']
        return {'y_prob': F.sigmoid(Y_pre)}




# Copyright (c) 2024, Tri Dao, Albert Gu.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class Mamba2Simple(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=128,
            d_conv=4,
            conv_init=None,
            expand=2,
            headdim=64,
            ngroups=1,
            A_init_range=(1, 16),
            dt_min=0.001,
            dt_max=0.1,
            dt_init_floor=1e-4,
            dt_limit=(0.0, float("inf")),
            learnable_init_states=False,
            activation="swish",
            bias=False,
            conv_bias=True,
            # Fused kernel and sharding options
            chunk_size=256,
            use_mem_eff_path=True,
            layer_idx=None,  # Absorb kwarg for general module
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states = repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out


def contains_sequences(tokenized_seq, tokenizer, sequences=["TTT"]):
    """
   Detect whether the sequence contains the specified list of sequences
  param tokenized_seq: The input sequence (tokenized)
  param sequences: The list of sequences to be detected
  return: Boolean value, indicating whether any of the specified sequences is included
    """

    for seq in sequences:
        if seq not in tokenizer.tkn2id:
            tokenizer.tkn2id[seq] = len(tokenizer.tkn2id)


    seq_ids = [tokenizer.tkn2id[seq] for seq in sequences]

    #Check whether tokenized_seq contains any of the specified sequence ids
    for seq_id in seq_ids:
        if seq_id in tokenized_seq:
            return True

    return False


class KGPDPLAM_alpha_Mamba2(nn.Module):
    def __init__(self, classNum, tknEmbedding=None, tknNum=None,
                 embSize=64, dkEnhance=1, freeze=False,
                 L=4, H=256, A=4, maxRelativeDist=7,
                 embDropout=0.2, hdnDropout=0.15, paddingIdx=-100, tokenizer=None):
        super(KGPDPLAM_alpha_Mamba2, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            tknEmbedding, freeze=freeze
        ) if tknEmbedding is not None else nn.Embedding(
            tknNum, embSize, padding_idx=paddingIdx
        )
        self.dropout1 = nn.Dropout(p=embDropout)

        self.conv1 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(p=embDropout)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.dropout3 = nn.Dropout(p=embDropout)


        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mamba': Mamba2Simple(
                    d_model=embSize,
                    d_state=128,
                    d_conv=4,
                    expand=2,
                    headdim=256,
                    activation="swish",
                ),
                'dropout': nn.Dropout(p=0.1),
                'ln': nn.LayerNorm(embSize)

            }) for _ in range(L)
        ])

        self.improved_deepPseudoLabelwiseAttn = Improved_DeepPseudoLabelwiseAttention(
            embSize, classNum, L=-1, hdnDropout=hdnDropout, dkEnhance=1,tokenizer=tokenizer, sequences=["TTT"]
        )
        self.fcLinear = MLP(embSize, 1)


        self.tokenizer = tokenizer


    def forward(self, data):

        x = data['tokenizedKgpSeqArr'] @ self.embedding.weight  # => batchSize × seqLen × embSize
        x = self.dropout1(x)


        x1 = self.conv1(x)
        x1 = self.dropout2(x1)

        x2 = self.conv2(x1)
        x2 = self.dropout3(x2)

        x = torch.cat((x, x1, x2), dim=1)

        for layer in self.layers:

            x = layer['mamba'](x)

            x = layer['dropout'](x)
            x = layer['ln'](x)
        pVec = torch.mean(x, dim=1)
        pVec = pVec / torch.sqrt(torch.sum(pVec ** 2, dim=1, keepdim=True))
        # # improvedDeepPseudoLabelwiseAttention
        x, attn = self.improved_deepPseudoLabelwiseAttn(x)

        x = self.fcLinear(x).squeeze(dim=2)
        return {'y_logit': x, 'p_vector': pVec, 'attn_weights': attn}






