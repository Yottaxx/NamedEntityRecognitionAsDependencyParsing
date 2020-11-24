import torch
import pandas as pd

def batch_computeF1(labels, preds, masks):
    scoreF1 = 0
    for i in range(len(labels)):
        label = labels[i]
        pred = preds[i]
        mask = masks[i]
        # cutting the padding
        true_len = int(mask.sum().item())
        pred = pred[:true_len, :true_len]
        label = label[:true_len, :true_len]
        scoreF1+=computeF1(label, pred)
    return scoreF1/len(labels)

def computeF1(label, pred):
    # 将label和pred读入，计算F1score返回
    label_items = get_entities(label, label=True)
    # use new func
    pred_items = Rm2entities(pred)
    # pred_items = get_entities(pred)
    label_num = len(label_items)
    pred_num = len(pred_items)
    # print("label_num:", label_num, "pred_num", pred_num)

    same_num = count_same_entities(label_items, pred_items)
    if same_num == 0:
        return 0

    precision = float(same_num)/float(label_num)
    recall = float(same_num)/float(pred_num)
    score = 2*precision*recall/(precision+recall)
    # print(score)
    return score

def get_entities(input_tensor, label=False):
    # Rm 为l*l*c的tensor，表征着每一个起始i终止j的片段属于各个实体的概率
    # 将该矩阵解码为实体列表，每一个实体用tuple表示，(category_id, pos_b, pos_e)
    # 如果lebel标签为True，则输入为l*l的tensor
    entities = []
    # 根据输入是否为label，都将输入矩阵转化为l*l的类别矩阵
    if not label:
        _, cate_tensor = clash_check(Rm=input_tensor)
    else:
        cate_tensor = input_tensor
    # 根据类别矩阵得到实体列表(cate_id, pos_b, pos_e)
    cate_indices = torch.nonzero(cate_tensor)
    for index in cate_indices:
        cate_id = cate_tensor[index[0], index[1]]
        entities.append((int(index[0]), int(index[1]), int(cate_id)))

    return entities

def count_same_entities(label_items, pred_items):
    count = 0
    for item in label_items:
        if item in pred_items:
            count+=1
    return count

def clash_check(Rm):
    #print(Rm.shape)
    # get socre and pred l*l
    score, cate_pred = Rm.max(dim=-1)
    #cate_pred_ = Rm.argmax(dim=-1)
    #print(cate_pred.sum().item(), cate_pred_.sum().item())

    # # mask the max score of max cate in Rm
    # max_mask = (Rm == score.unsqueeze(-1))
    # Rm = torch.where(max_mask, torch.zeros_like(Rm), Rm)
    #
    # # recompute the max, means to select the 2ed max category
    # score, cate_pred = Rm.max(-1)

    # mask category of non-entity
    zero_mask = (cate_pred==torch.tensor([0]).to(score.device))
    score = torch.where(zero_mask, torch.zeros_like(score), score)

    if score.sum() > 0:
        print("Sum of categoriesID: ", cate_pred.sum().item(), "Sum of p: ", score.sum().item())

    # pos_b <= pos_e check
    score = torch.triu(score)
    cate_pred = torch.triu(cate_pred)
    # start pos clash check
    max_in_row, idx_in_row = score.max(dim = 1)
    score = torch.where((score>=max_in_row.unsqueeze(-1)), score, torch.zeros_like(score))
    cate_pred = torch.where((score>=max_in_row.unsqueeze(-1)), cate_pred, torch.zeros_like(cate_pred))
    # end pos clash check
    max_in_col, idx_in_col = score.max(dim = 0)
    score = torch.where((score>=max_in_col.unsqueeze(0)), score, torch.zeros_like(score))
    cate_pred = torch.where((score>=max_in_col.unsqueeze(0)), cate_pred, torch.zeros_like(cate_pred))
    return score, cate_pred

# new clash check func, from Rm matrix to entities set
def Rm2entities(Rm, is_flat_ner=True):
    # get socre and pred l*l
    score, cate_pred = Rm.max(dim=-1)

    # filter mask
    # mask category of non-entity
    zero_mask = (cate_pred == torch.tensor([0]).float().to(score.device))
    score = torch.where(zero_mask, torch.zeros_like(score), score)
    # pos_b <= pos_e check
    score = torch.triu(score) # l*l
    cate_pred = torch.triu(cate_pred) # l*l

    # get all entity list
    all_entity = []
    score_shape = score.shape
    score = score.reshape(-1)
    cate_pred = cate_pred.reshape(-1)
    entity_indices = (score != 0).nonzero(as_tuple = False).squeeze(-1)
    for i in entity_indices:
        i = int(i)
        score_i = float(score[i].item())
        cate_pred_i = int(cate_pred[i].item())
        pos_s = i//score_shape[0]
        pos_e = i%score_shape[0]
        all_entity.append((pos_s, pos_e, cate_pred_i, score_i))
    # sort by score
    all_entity = sorted(all_entity, key=lambda x:x[-1])
    res_entity = []
    for ns,ne,t,_ in all_entity:
        for ts,te,_ in res_entity:
            if ns < ts <= ne < te or ts < ns <= te < ne:
                # for both nested and flat ner no clash is allowed
                break
            if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                # for flat ner nested mentions are not allowed
                break
        else:
            res_entity.append((ns, ne, t))
    return set(res_entity)

def entities2df(entities):
    df = pd.DataFrame.from_records(entities, columns=['Category', 'Pos_s', 'Pos_e', 'Privacy'])
    ID_list = list(range(len(df)))
    for i in ID_list:
        i = str(i)
    df.insert(0, "ID", ID_list)
    return df

# loss compute problem
# before loss compute, filter some useful pos in out & label
def get_useful_ones(out, label, attention_mask):
    # get mask, mask the padding and down triangle
    attention_mask = attention_mask.unsqueeze(1).expand(-1, attention_mask.shape[-1], -1)
    attention_mask = torch.triu(attention_mask)
    # flatten
    mask = attention_mask.reshape(-1)
    tmp_out = out.reshape(-1, out.shape[-1])
    tmp_label = label.reshape(-1)
    # index select, for gpu speed
    indices = mask.nonzero(as_tuple=False).squeeze(-1).long()
    tmp_out = tmp_out.index_select(0, indices)
    tmp_label = tmp_label.index_select(0, indices)

    return tmp_out, tmp_label

if __name__ == '__main__':
    labels = torch.randn(3,5,5)
    pred = torch.randn(3,5,5,3)
    mask = torch.ones(3, 5)
    #print(batch_computeF1(labels, pred, mask))