from torch import nn as nn
from torch.nn import functional as F
import torch,time,os,random
import numpy as np
from collections import OrderedDict
from math import floor


###改进的定位注意力模块
class Improved_DeepPseudoLabelwiseAttention(nn.Module):
    def __init__(self, inSize, classNum, L=1, M=64, hdnDropout=0.1, actFunc=nn.ReLU, dkEnhance=4, recordAttn=True,
                 name='DPLA', tokenizer=None, sequences=["TTT"], enhanceFactor=1.5):
        super(Improved_DeepPseudoLabelwiseAttention, self).__init__()
        if L > -1:
            self.inLWA = nn.Linear(inSize, M)

            hdnLWAs, hdnFCs, hdnBNs, hdnActFuncs = [], [], [], []
            for i in range(L):
                hdnFCs.append(nn.Linear(inSize, inSize))
                hdnBNs.append(nn.BatchNorm1d(inSize))
                hdnActFuncs.append(actFunc())
                hdnLWAs.append(nn.Linear(inSize, M))
            self.hdnLWAs = nn.ModuleList(hdnLWAs)
            self.hdnFCs = nn.ModuleList(hdnFCs)
            self.hdnBNs = nn.ModuleList(hdnBNs)
            self.hdnActFuncs = nn.ModuleList(hdnActFuncs)

        self.outFC = nn.Linear(inSize, inSize * dkEnhance)
        self.outBN = nn.BatchNorm1d(inSize * dkEnhance)
        self.outActFunc = actFunc()
        self.outLWA = nn.Linear(inSize * dkEnhance, classNum)

        self.dropout = nn.Dropout(p=hdnDropout)
        self.name = name
        self.L = L

        self.recordAttn = recordAttn
        self.tokenizer = tokenizer
        self.sequences = sequences  # 要增强注意力的序列
        self.enhanceFactor = enhanceFactor  # 增强指定序列注意力的因子

    def forward(self, x):
        # x: batchSize × seqLen × inSize
        all_attn = []  # 保存所有注意力分布

        # 创建一个mask来标记哪些序列包含 'TTT' 或其他指定序列
        contains_sequence_mask = self._contains_sequences_mask(x)  # 获取包含目标序列的mask

        if self.L > -1:
            # input layer
            score = self.inLWA(x)  # => batchSize × seqLen × M
            alpha = self.dropout(F.softmax(score, dim=1))  # => batchSize × seqLen × M

            # 保存第1层注意力
            all_attn.append(alpha.clone().detach().cpu())

            # 增强指定序列的注意力
            alpha = self._enhance_attention(alpha, contains_sequence_mask)

            a_nofc = alpha.transpose(1, 2) @ x  # => batchSize × M × inSize

            # hidden layers
            score = 0
            for i, (lwa, fc, bn, act) in enumerate(zip(self.hdnLWAs, self.hdnFCs, self.hdnBNs, self.hdnActFuncs)):
                a = fc(a_nofc)  # => batchSize × M × inSize
                a = bn(a.transpose(1, 2)).transpose(1, 2)  # => batchSize × M × inSize
                a_pre = self.dropout(act(a))  # => batchSize × M × inSize

                score = lwa(a_pre)  # + score
                alpha = self.dropout(F.softmax(score, dim=1))

                # 保存隐藏层注意力
                all_attn.append(alpha.clone().detach().cpu())

                # 增强指定序列的注意力
                alpha = self._enhance_attention(alpha, contains_sequence_mask)

                a_nofc = alpha.transpose(1, 2) @ a_pre + a_nofc  # => batchSize × M × inSize

            a_nofc = self.dropout(a_nofc)
        else:
            a_nofc = x

        # output layers
        if self.L > -1:
            a = self.outFC(a_nofc)  # => batchSize × M × inSize
            a = self.outBN(a.transpose(1, 2)).transpose(1, 2)  # => batchSize × M × inSize
            a = self.dropout(self.outActFunc(a))  # => batchSize × M × inSize
        else:
            a = a_nofc

        score = self.outLWA(a)  # => batchSize × M × classNum
        alpha = self.dropout(F.softmax(score, dim=1))  # => batchSize × M × classNum

        # 保存输出层注意力
        all_attn.append(alpha.clone().detach().cpu())

        # 增强指定序列的注意力
        alpha = self._enhance_attention(alpha, contains_sequence_mask)

        x = alpha.transpose(1, 2) @ a  # => batchSize × classNum × inSize

        return x, all_attn  # 返回最终输出和所有注意力分布

    def _contains_sequences_mask(self, x):
        """
        根据输入的 tokenized 序列生成一个 mask，表示哪些序列包含指定的目标序列。
        :param x: 输入的 tokenized 序列，形状为 batchSize × seqLen
        :return: mask，表示哪些序列包含目标序列，形状为 batchSize × seqLen
        """
        batch_size, seq_len = x.shape[:2]
        mask = torch.zeros(batch_size, seq_len).to(x.device)  # 创建一个零矩阵作为 mask

        for i in range(batch_size):
            tokenized_seq = x[i]  # 获取当前样本的 tokenized 序列
            if self.contains_sequences(tokenized_seq):  # 检查是否包含目标序列
                mask[i] = 1  # 如果包含目标序列，设置对应的 mask 为 1

        return mask

    def contains_sequences(self, tokenized_seq):
        """
        检测序列中是否包含指定的序列列表
        :param tokenized_seq: 输入的序列 (tokenized)
        :return: 布尔值，表示是否包含任何一个指定的序列
        """
        # 确保每个序列都在 tokenizer.tkn2id 中
        for seq in self.sequences:
            if seq not in self.tokenizer.tkn2id:
                self.tokenizer.tkn2id[seq] = len(self.tokenizer.tkn2id)  # 为每个序列分配一个新的 token ID

        # 获取所有序列的 token ID
        seq_ids = [self.tokenizer.tkn2id[seq] for seq in self.sequences]

        # 检查 tokenized_seq 是否包含任何一个指定的序列 ID
        for seq_id in seq_ids:
            if seq_id in tokenized_seq:
                return True  # 如果匹配到任何一个序列，则返回 True

        return False  # 如果所有序列都未匹配，则返回 False

    def _enhance_attention(self, alpha, mask):
 
        enhance_factor = self.enhanceFactor
        enhanced_alpha = alpha * (1 + mask.unsqueeze(2) * (enhance_factor - 1))  # 增强目标序列位置的注意力
        return enhanced_alpha



