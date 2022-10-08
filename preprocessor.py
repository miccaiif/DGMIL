from tqdm import tqdm
import os
import numpy as np
import sklearn.metrics as skm
import heapq
import random

#for extreme samples preprocessing
def flatten_feats_and_name(path):
    all_slide_all_patch_feats = []
    all_slide_all_patch_name_list = []
    for slide in tqdm(os.listdir(path)):
        slide_path = path+slide
        slide_feats = np.load(slide_path)
        all_slide_all_patch_feats.append(slide_feats)
        for idx,_ in enumerate(slide_feats):
            patch_name = slide.split('.')[0] + '_' + str(idx)
            all_slide_all_patch_name_list.append(patch_name)
    all_slide_all_patch_feats_ = [b for a in all_slide_all_patch_feats for b in a]
    return all_slide_all_patch_feats_, all_slide_all_patch_name_list


def sort_all_feats_and_pick_forpos(feats,namelist): #for pos slide largest index
    num = int(len(feats) * 0.1)
    #num = 5776 #10% for camelyon
    # Find the three largest indexes/ nsmallest is the opposite of nlargest, find the smallest
    index = list(map(feats.index, heapq.nlargest(num, feats)))
    # feats = np.array(feats)
    namelist = np.array(namelist)
    patch_pick_name_list = namelist[index]
    return patch_pick_name_list


def sort_all_feats_and_pick_forneg(feats,namelist): #for neg slide smallest index
    num = int(len(feats) * 0.1)
    # num = 5776 10% for camelyon
    # Find the three largest indexes/ nsmallest is the opposite of nlargest, find the smallest
    index = list(map(feats.index, heapq.nsmallest(num, feats)))
    # feats = np.array(feats)
    namelist = np.array(namelist)
    patch_pick_name_list = namelist[index]
    return patch_pick_name_list


def sort_and_generate_feats_bank_forpos_dynamic(dis,feats): #for pos slide largest index
    num = int(len(feats) * 0.1)
    # num = 5776 10% for camelyon
    dis = dis.tolist()
    # Find the three largest indexes/ nsmallest is the opposite of nlargest, find the smallest
    index = list(map(dis.index, heapq.nlargest(num, dis)))
    # feats = np.array(feats)
    pos_patch_feats_bank = feats[index]
    return pos_patch_feats_bank


def sort_and_generate_feats_bank_forneg_dynamic(dis,feats): #for neg slide smallest index
    num = int(len(feats) * 0.1)
    # num = 5776 10% for camelyon
    dis = dis.tolist()
    # Find the three largest indexes/ nsmallest is the opposite of nlargest, find the smallest
    index = list(map(dis.index, heapq.nsmallest(num, dis)))
    # feats = np.array(feats)
    neg_patch_feats_bank = feats[index]
    return neg_patch_feats_bank


def sort_and_generate_feats_bank_forpos_dynamic_for_look(dis,feats,patch_label):  #only for checking pos slides
    num = int(len(feats) * 0.1)
    # num = 5776 10% for camelyon
    dis = dis.tolist()
    index = list(map(dis.index, heapq.nlargest(num, dis)))
    # feats = np.array(feats)
    pos_patch_feats_bank = feats[index]
    pos_label_pick = patch_label[index]
    return pos_patch_feats_bank, pos_label_pick


def sort_and_generate_feats_bank_forneg_dynamic_for_look(dis,feats):  #only for checking neg slides
    num = int(len(feats) * 0.1)
    # num = 5776 10% for camelyon
    dis = dis.tolist()
    index = list(map(dis.index, heapq.nsmallest(num, dis)))
    # feats = np.array(feats)
    neg_patch_feats_bank = feats[index]
    return neg_patch_feats_bank


def generate_fintuning_feats_bank(path,namelist):
    patch_feats_bank = np.zeros(512)
    for name in namelist:
        patch_idx = int(name.split('_')[-1])
        # print(name)
        slide_name = path + name[0:name.rfind('_' + name.split('_')[-1])] + '.np.npy'
        patch_feats = np.load(slide_name)[patch_idx][1:]
        patch_feats_bank = np.vstack((patch_feats_bank,patch_feats))
    return patch_feats_bank[1:,:]


def main_generator(dis_path_neg,dis_path_pos,org_path_neg,org_path_pos):
    all_slide_all_patch_feats_dis_neg, all_slide_all_patch_name_list_neg = flatten_feats_and_name(dis_path_neg)
    all_slide_all_patch_feats_dis_pos, all_slide_all_patch_name_list_pos = flatten_feats_and_name(dis_path_pos)
    neg_patch_pick_namelist = sort_all_feats_and_pick_forpos(all_slide_all_patch_feats_dis_neg,
                                                             all_slide_all_patch_name_list_neg)
    pos_patch_pick_namelist = sort_all_feats_and_pick_forneg(all_slide_all_patch_feats_dis_pos,
                                                             all_slide_all_patch_name_list_pos)
    fintuning_feats_bank_neg = generate_fintuning_feats_bank(org_path_neg, neg_patch_pick_namelist)
    fintuning_feats_bank_pos = generate_fintuning_feats_bank(org_path_pos, pos_patch_pick_namelist)
    fintuning_feats = np.vstack((fintuning_feats_bank_neg, fintuning_feats_bank_pos))
    label = np.array([0] * len(fintuning_feats_bank_neg) + [1] * len(fintuning_feats_bank_pos))
    return fintuning_feats, label


def main_generator_dynamic(dis_neg_train, new_neg_feats, dis_pos_train, new_pos_feats):

    neg_patch_feats_bank = sort_and_generate_feats_bank_forneg_dynamic(dis_neg_train, new_neg_feats)
    pos_patch_feats_bank = sort_and_generate_feats_bank_forpos_dynamic(dis_pos_train,new_pos_feats)

    fintuning_feats = np.vstack((neg_patch_feats_bank, pos_patch_feats_bank))
    label = np.array([0] * len(neg_patch_feats_bank) + [1] * len(pos_patch_feats_bank))
    return fintuning_feats, label


def main_generator_dynamic_for_look(dis_neg_train, new_neg_feats, dis_pos_train, new_pos_feats, patch_label):

    neg_patch_feats_bank = sort_and_generate_feats_bank_forneg_dynamic_for_look(dis_neg_train, new_neg_feats) #only look pos slides
    pos_patch_feats_bank, pos_label_picklist = sort_and_generate_feats_bank_forpos_dynamic_for_look(dis_pos_train,new_pos_feats, patch_label)#only look pos slides

    pos_ratio = np.sum(np.array(pos_label_picklist))/len(pos_label_picklist)

    fintuning_feats = np.vstack((neg_patch_feats_bank, pos_patch_feats_bank))
    label = np.array([0] * len(neg_patch_feats_bank) + [1] * len(pos_patch_feats_bank))
    return fintuning_feats, label, pos_ratio


def main_MAE_generator(dis_training_neg_path,dis_training_pos_path,feats_training_neg_path,feats_training_pos_path):
    dis_training_neg = np.load(dis_training_neg_path)
    dis_training_pos = np.load(dis_training_pos_path)
    feats_training_neg = np.load(feats_training_neg_path)
    feats_training_pos = np.load(feats_training_pos_path)

    fintuning_feats, label = main_generator_dynamic(dis_training_neg, feats_training_neg, dis_training_pos, feats_training_pos)

    return fintuning_feats, label


def main_MAE_generator_for_look(dis_training_neg_path,dis_training_pos_path,feats_training_neg_path,feats_training_pos_path, patch_label_pos_path):
    dis_training_neg = np.load(dis_training_neg_path)
    dis_training_pos = np.load(dis_training_pos_path)
    feats_training_neg = np.load(feats_training_neg_path)
    feats_training_pos = np.load(feats_training_pos_path)
    patch_label_training_pos = np.load(patch_label_pos_path)

    fintuning_feats, label, pos_ratio = main_generator_dynamic_for_look(dis_training_neg, feats_training_neg, dis_training_pos, feats_training_pos, patch_label_training_pos)

    return fintuning_feats, label, pos_ratio #only look pos slides


#Randomly select top 10% samples during initialization
def main_topk_generator(feats_training_neg_path,feats_training_pos_path):
    feats_training_neg = np.load(feats_training_neg_path)
    feats_training_pos = np.load(feats_training_pos_path)
    num_list_for_neg = random.sample(range(0, len(feats_training_neg[:,0])), 5776) #5776 is 10% of Camelyon16
    num_list_for_pos = random.sample(range(0, len(feats_training_pos[:, 0])), 5776) #5776 is 10% of Camelyon16
    neg_patch_feats_bank = feats_training_neg[num_list_for_neg]
    pos_patch_feats_bank = feats_training_pos[num_list_for_pos]
    fintuning_feats = np.vstack((neg_patch_feats_bank, pos_patch_feats_bank))
    label = np.array([0] * len(neg_patch_feats_bank) + [1] * len(pos_patch_feats_bank))

    return fintuning_feats, label

