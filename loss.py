import torch
device = torch.device('cuda')


def true_len(t):
    len = 0
    for num in t :
        if not torch.eq(num, torch.tensor(0)):
            len+=1
    return len


def monotonic_alignment(p):
    bsz, tgt_len, src_len = p.size()
        
    p_ext = p.roll(1, [-1]).unsqueeze(-2).expand(-1, -1, src_len, -1).triu(1)

    T = (1 - p_ext).cumprod(-1).triu()

    alpha = [torch.bmm(p[:, [0]], T[:, [0]].squeeze(dim=1))]

    for i in range(1, tgt_len):
        alpha.append(p[:, [i]] * torch.bmm(alpha[i - 1], T[:, [i]].squeeze(dim=1)))
    return torch.cat(alpha, dim=1)




def varianceLoss(alpha):
    #alpha size: bsz, tgt_len, src_len
    alpha = alpha.to(device)
    bsz, tgt_len, src_len = alpha.size()
    k = torch.FloatTensor([(i+1) for i in range(src_len)])
    k = k.expand(tgt_len, src_len).to(device)
    k_square = torch.square(k).to(device)
    ej2i = torch.sum(k_square*alpha, dim=-1, keepdim=True)
    eji_2 = torch.square(torch.sum(k*alpha, dim=-1, keepdim=True))
    return torch.mean(ej2i-eji_2)


def DAL_loss(alpha, x_len, y_len, tgt_padding, num_haed, num_layer):
    alpha = alpha.to(device)
    x_len = x_len.to(device)
    y_len = y_len.to(device)
    tgt_padding = tgt_padding.to(device)

    bsz_head_layer, tgt_len, src_len = alpha.size()
    k = torch.FloatTensor([(i+1) for i in range(src_len)])
    k = k.expand(tgt_len, src_len).to(device)
    g_i = alpha*k
    g = torch.sum(g_i, dim=-1)
    # g = g.transpose(-1, -2)
    # assume batch1:|x|=5,|y|=2 // batch2:|x|=4,|y|=3
    # x_len = torch.FloatTensor([5, 4])
    # y_len = torch.FloatTensor([2, 3])
    gamma = torch.div(x_len, y_len)
    # print('gamma:', gamma)
    gamma = [gamma_i.expand(num_haed*num_layer)  for gamma_i in gamma]
    gamma = torch.cat(gamma, dim=0)
    # for i in range(1, g.size(0)):
    #     g[i] = torch.maximum(g[i], g[i-1]+gamma)
    # g = g.transpose(-1, -2)
    
    padding_mask = torch.cat([padding_mask_i.unsqueeze(0).expand(num_haed*num_layer, padding_mask_i.size(0)) for padding_mask_i in tgt_padding], dim=0)
    
    ### simulate 1/|y|
    one_divide_y = torch.cat([l_y.expand(num_haed*num_layer, 1) for l_y in y_len], dim=0).squeeze(-1)
    
    ### calculate DAL
    i_1 = torch.FloatTensor([i for i in range(tgt_len)])
    i_1 = i_1.expand(bsz_head_layer, tgt_len).to(device)
    i_1_gamma = i_1 * gamma.unsqueeze(1).expand(bsz_head_layer, tgt_len)
    total = g - i_1_gamma
    total = total * padding_mask
    total = torch.sum(total, dim=-1)
    DAL = torch.div(total, one_divide_y)
    return torch.mean(DAL)

def isNAN_isINF(tensor):
    flag = False
    if(torch.isnan(tensor).any()):
        print('============== NAN detect ==============')
        flag = True
    if(torch.isinf(tensor).any()):
        print('============== INF detect ==============')
        flag = True
    return flag
    