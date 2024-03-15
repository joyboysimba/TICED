import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import Counter  
import math
import numpy as np
import pandas as pd
from scipy import stats



class TICED(nn.Module):
    """

    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        position_embed_dim/embedding_dim(int): the dimension of item embedding
        n_layers(int): the number of gru layers

    """

    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, max_len, position_embed_dim, lambda_denoise, num_heads, pos_num,neighbor_num, n_layers=1):
        super(TICED, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.position_embed_dim = position_embed_dim
        self.lambdas = lambda_denoise
        self.pos_num = pos_num
        self.neighbor_num = neighbor_num
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.emb1 = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.multihead_attn = nn.MultiheadAttention(self.embedding_dim,num_heads)


        self.position_emb = nn.Embedding(self.pos_num, self.position_embed_dim, padding_idx=0)
        self.position_dropout = nn.Dropout(0.3)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout70 = nn.Dropout(0.70)

        # batchnormalization
        self.bn = torch.nn.BatchNorm1d(max_len, affine=False)
        self.bn1 = torch.nn.BatchNorm1d(embedding_dim, affine=False)
        
        self.final2std_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std_last = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v_h2std = nn.Linear(self.embedding_dim, 1, bias=True)

        self.final2std2_std = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std2_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.final2std3_std = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std3_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)

        
        self.user2item_dim = nn.Linear(self.hidden_size, self.embedding_dim, bias=True)
        self.pos2item_dim = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        # Dual gating mechanism
        self.w_u_p = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_u_r = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_u = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.u_u_p = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.u_u_r = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.u_u = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.u_u_h = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.u_u_e = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_p_c = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.u_p_h = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.u_p_e = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.u_p_c = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.merge_n_c = nn.Linear(self.embedding_dim * 4, self.embedding_dim, bias=True)
        # attention to initial
        self.v1_w = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v1 = nn.Linear(self.embedding_dim, 1, bias=True)
        self.sf = nn.Softmax()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    def forward(self, seq, lengths):
        # get original item embedding
        embs_origin = self.emb(seq)  # seq:(19,bs)
        lengths_ori = lengths
        embs_origin = embs_origin.permute(1, 0, 2) # (bs,19,es)

        # get position embedding  pos+length
        item_position = torch.tensor(range(1, seq.size()[0]+1), device=self.device) # [19]
        item_position = item_position.unsqueeze(1).expand_as(seq).permute(1, 0) # (bs,19)
        len_d = torch.Tensor(lengths_ori).unsqueeze(1).expand_as(seq.permute(1, 0))*5-1 # (bs,19) 长度的权重
        len_d = len_d.type(torch.LongTensor).cuda()
        mask_position = torch.where(seq.permute(1, 0) > 0, torch.tensor([1], device=self.device),
                           torch.tensor([0], device=self.device)) # (bs,19)
        
        item_imp = torch.zeros_like(seq)
        for i in range(seq.shape[1]):
            unique_values, counts = torch.unique(seq[:, i], return_counts=True)
            item_imp[:, i] = counts[torch.searchsorted(unique_values, seq[:, i])]

        #item_imp = self.emb1(item_imp * mask_position.permute(1, 0)) #(19,bs,64)
        #item_position = item_position * mask_position * len_d     # (bs,19)
        item_position = item_imp.permute(1, 0) * mask_position 
        
        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device=self.device),
                           torch.tensor([0.], device=self.device)) # 浮点mask (bs,19)

        hidden = self.init_hidden(seq.size(1)) #(bs,hiddensize)
        #embs_origin(19,256,64),19是规定的最大长度，length表示每一个会话的真实长度<19
        embs_padded = pack_padded_sequence(embs_origin.permute(1, 0, 2) , lengths) 
        gru_out, hidden = self.gru(embs_padded, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out) # get user embeding gru_out (real_lenth,bs,hiddensize)
        now_emb = hidden[-1] # 当前长度的时间信息(bs,hiddensize) 
        glo_emb = torch.sum(gru_out, 0) #全局信息 real_lenth,bs,hiddensize)->(bs,hiddensize)

        pos_embs = self.position_dropout(self.position_emb(item_position)) #torch.Size([bs, 19, posdim])
        pos_embs = self.dropout30(self.pos2item_dim(pos_embs)) # torch.Size([bs, 19, 64])
        usernow_emb = self.dropout30(self.user2item_dim(now_emb)).unsqueeze(1).expand_as(embs_origin) 
        user_emb_expand = self.dropout30(self.user2item_dim(glo_emb)).unsqueeze(1).expand_as(embs_origin)

        user_p = torch.sigmoid(self.w_u_p(user_emb_expand)+self.u_u_p(usernow_emb))
        user_r = torch.sigmoid(self.w_u_r(user_emb_expand) + self.u_u_r(embs_origin))  #torch.Size([bs, 19, 64])
        uw_emb_h = self.tanh(self.w_u(user_emb_expand) + self.u_u(user_p*embs_origin))
        uw_emb = user_p * user_emb_expand + user_r * uw_emb_h


        fre_h = torch.sigmoid(self.u_u_h(uw_emb) + self.u_p_h(pos_embs))
        fre_e = torch.sigmoid(self.u_u_e(uw_emb) + self.u_p_e(pos_embs))
        pw_emb_h = self.tanh(self.w_p_c(uw_emb) + self.u_p_c(pos_embs))
        embs_final = fre_h*user_emb_expand + fre_e*pw_emb_h

        # batchnormalization
        embs_final = self.bn(embs_final) #torch.Size([bs, 19, 64])

        ini_emb = self.final2std_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())
        ini_atten = self.v1_w(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())
        ini_atten = self.tanh(ini_atten)  #torch.Size([bs, 19, 64])
        ini_w = self.v1(ini_atten).expand_as(ini_emb) * ini_emb  #torch.Size([bs, 19, 64])

        
        layer1mask = mask.unsqueeze(2).expand_as(ini_emb) * ini_w   #torch.Size([bs, 19, 64])
        layer1mask = torch.sum(layer1mask, 1)  #torch.Size([bs, 64])
        layer1mask = mask.unsqueeze(2).expand_as(ini_emb) * layer1mask.unsqueeze(1).expand_as(ini_emb)


        output1, attn_weights1 = self.multihead_attn(embs_final, ini_emb, layer1mask)
        layer1emb =  torch.sum(attn_weights1, 1).permute(1,0).unsqueeze(2).expand_as(embs_final) * embs_final

        beta1 = self.v_h2std(torch.sigmoid(output1 + layer1emb).view(-1, self.embedding_dim)).view(
            mask.size()) # torch.Size([bs, 19])
        #  beta1 = self.v_h2std(torch.sigmoid(ini_emb + layer1mask ).view(-1, self.embedding_dim)).view(mask.size())

        # item filter
        beta1 = self.sf(beta1)
        beta1_v = torch.mean(beta1, 1, True)[0].expand_as(beta1) # torch.Size([bs, 19])
        beta1_mask = beta1 - self.lambdas * beta1_v    #torch.Size([bs, 19])

        beta1 = torch.where(beta1_mask > 0, beta1,
                           torch.tensor([0.], device=self.device))

        sess_std = torch.sum(beta1.unsqueeze(2).expand_as(embs_final) * embs_final, 1)

        sess_std = self.dropout70(sess_std) # torch.Size([bs, 64])

        layer2cur = self.final2std2_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size()) #torch.Size([bs, 19, 64])
        layer2_a = self.final2std2_std(sess_std) #torch.Size([bs, 64])
        layer2_a_expand = layer2_a.unsqueeze(1).expand_as(layer2cur) #torch.Size([bs, 19, 64])
        layer2mask = mask.unsqueeze(2).expand_as(layer2cur) * layer2_a_expand #torch.Size([bs, 19, 64])


        output2, attn_weights2 = self.multihead_attn(embs_final, layer2cur, layer2mask)
        layer2emb =  torch.sum(attn_weights2, 1).permute(1,0).unsqueeze(2).expand_as(embs_final) * embs_final

        beta2 = self.v_h2std(torch.sigmoid(output2 + layer2emb).view(-1, self.embedding_dim)).view(
            mask.size()) # torch.Size([bs, 19])

        beta2 = self.sf(beta2) #torch.Size([bs, 19])
        beta2_v = torch.mean(beta2, 1, True)[0].expand_as(beta2)
        beta2_mask = beta2 - self.lambdas * beta2_v

        beta2 = torch.where(beta2_mask > 0, beta2,
                            torch.tensor([0.], device=self.device))
        sess_std2 = torch.sum(beta2.unsqueeze(2).expand_as(embs_final) * embs_final, 1)
        sess_std2 = self.dropout30(sess_std2)  #torch.Size([bs, 64])

        layer3cur = self.final2std3_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())
        layer3_a = self.final2std3_std(sess_std2)
        layer3_a_expand = layer3_a.unsqueeze(1).expand_as(layer3cur)
        layer3mask = mask.unsqueeze(2).expand_as(layer3cur) * layer3_a_expand


        output3, attn_weights3 = self.multihead_attn(embs_final, layer3cur, layer3mask)
        layer3emb = torch.sum(attn_weights3, 1).permute(1,0).unsqueeze(2).expand_as(embs_final)* embs_final

        beta3 = self.v_h2std(torch.sigmoid(output3 + layer3emb).view(-1, self.embedding_dim)).view(
            mask.size()) # torch.Size([bs, 19])


        beta3 = self.sf(beta3)
        beta3_v = torch.mean(beta3, 1, True)[0].expand_as(beta3)
        beta3_mask = beta3 - self.lambdas * beta3_v

        beta3 = torch.where(beta3_mask > 0, beta3,
                            torch.tensor([0.], device=self.device)) # torch.Size([bs, 19])
        sess_std3 = torch.sum(beta3.unsqueeze(2).expand_as(embs_final) * embs_final, 1) 

        sess_current = sess_std3 # torch.Size([bs, 64])

        
        # cosine similarity
        sess_norm = F.normalize(sess_current, p=2, dim=1)
        sim_matrix = F.cosine_similarity(sess_norm.unsqueeze(1), sess_norm.unsqueeze(0), dim=2) #torch.Size([bs, bs])

        k_v = self.neighbor_num
        if sim_matrix.size()[0] < k_v:
            k_v = sim_matrix.size()[0]
        cos_topk, topk_indice = torch.topk(sim_matrix, k=k_v, dim=1) #torch.Size([bs, 5])
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_current[topk_indice] # torch.Size([bs, 5, 64])

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.embedding_dim) # torch.Size([bs, 5, 64])

        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess) #torch.Size([bs, 64])
        sess_final = torch.cat(
            [sess_current, neighbor_sess, sess_current + neighbor_sess, sess_current * neighbor_sess], 1)

        sess_final = self.dropout30(sess_final)
        sess_final = self.merge_n_c(sess_final)

        item_embs = self.emb(torch.arange(self.n_items).to(self.device))

        item_embs = self.bn1(item_embs)
        scores = torch.matmul(sess_final, item_embs.permute(1, 0))
        return scores  # torch.Size([bs, 43097])

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def transpose_for_scores(self, x, attention_head_size):
        # INPUT:  x'shape = [bs, seqlen, hid_size] hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  
        return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]
