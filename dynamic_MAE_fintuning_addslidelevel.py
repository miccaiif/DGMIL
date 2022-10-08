import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from preprocessor import main_MAE_generator, main_generator_dynamic
from model import Linear_projection_MAE
from pos_score_calculator import get_score_and_dis_MAE, get_score_and_dis_feats, get_auc_score
from tensorboardX import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description="DG-MIL main")

parser.add_argument("--exp-name", type=str, default="DG-MIL fintuning and test per epoch")
parser.add_argument("--dis_training_neg", type=str, default='./MAE_dynamic_trainingneg_dis.npy')
parser.add_argument("--dis_training_pos", type=str, default='./MAE_dynamic_trainingpos_dis.npy')
parser.add_argument("--feats_training_neg", type=str, default='./MAE_dynamic_trainingneg_feats.npy')
parser.add_argument("--feats_training_pos", type=str, default='./MAE_dynamic_trainingpos_feats.npy')
# parser.add_argument("--neg_dir_training", type=str, default='./Cam16_training_neg_features.npy')
# parser.add_argument("--pos_dir_training", type=str, default='./Cam16_training_pos_features.npy')
parser.add_argument("--neg_dir_testing", type=str, default='./MAE_testing_neg_feats.npy')
parser.add_argument("--pos_dir_testing", type=str, default='./MAE_testing_pos_feats.npy')

parser.add_argument("--model_save_dir", type=str, default= './MAE_dynamic_fintuning/')
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--num_cluster", type=int, default=10)
parser.add_argument("--seed", type=int, default=2022)
parser.add_argument('--summary_name', type=str, default='DGMIL_MAE_dynamic_cluster10_')

#for slide testing
parser.add_argument("--testing_feats_original", type=str, default='./test_MAE_feats.npy')
parser.add_argument("--num_bag_list_index", type=str, default='./num_bag_list_index.npy')
parser.add_argument("--test_slide_label", type=str, default='./test_slide_label.npy')

args = parser.parse_args()

writer = SummaryWriter(comment=args.summary_name)

if not os.path.isdir(args.model_save_dir):
    os.mkdir(args.model_save_dir)

def print_(loss):
    print ("The loss calculated: ", loss)

def model_train(train_feat,train_label,model,loss_fn,optimizer):
    model.train()
    y_pred = model(train_feat)
    loss = loss_fn(y_pred, train_label)
    print_(loss.item())

    # Zero gradients
    optimizer.zero_grad()
    loss.backward()  # Gradients
    optimizer.step()  # Update
    return loss.item()

def model_test(model,val_feat,labels_test):
    model.eval()
    pred = model(val_feat)
    pred = pred.detach().numpy()
    print("The accuracy of extreme samples test set is", accuracy_score(labels_test, np.argmax(pred, axis=1)))
    print("The auc of extreme samples test set is", roc_auc_score(labels_test, np.argmax(pred, axis=1)))
    return accuracy_score(labels_test, np.argmax(pred, axis=1)), roc_auc_score(labels_test, np.argmax(pred, axis=1))

def model_newfeats_extract(model,feats):
    model.eval()
    new_feats = model.projection_head(feats)
    return new_feats

def change_format_for_feats(feats):
    feats = torch.from_numpy(feats.astype(np.float32))
    return feats

def change_format_for_labels(labels):
    labels = torch.from_numpy(labels.astype(np.compat.long)).long()
    return labels


if __name__ == "__main__":
    device = "cuda:0"

    # loading extreme training samples based on distance (original MAE feats space), note that the feats are not dis,
    # but are picked based on dis (extreme samples) i.e. MAE feats space initialization

    fintuning_feats, label = main_MAE_generator(args.dis_training_neg, args.dis_training_pos, args.feats_training_neg, args.feats_training_pos)
    features_train, features_test, labels_train, labels_test = train_test_split(fintuning_feats, label, test_size=0.1,random_state=12345,
                                                                                shuffle=True)

    # change format for training and testing

    train_feat = change_format_for_feats(features_train)
    train_label = change_format_for_labels(labels_train)
    val_feat = change_format_for_feats(features_test)
    val_label = change_format_for_labels(labels_test)

    # loading original testing samples directly from orginal patch features

    neg_feats_training = np.load(args.feats_training_neg)
    pos_feats_training = np.load(args.feats_training_pos)
    neg_feats_testing = np.load(args.neg_dir_testing)
    pos_feats_testing = np.load(args.pos_dir_testing)

    # change format for directly testing all original feats

    all_test_original_feats = np.vstack((neg_feats_testing, pos_feats_testing))
    all_test_original_label = np.array(
        [0] * len(neg_feats_testing) + [1] * len(pos_feats_testing))

    all_test_original_feats = change_format_for_feats(all_test_original_feats)
    all_test_original_label = change_format_for_labels(all_test_original_label)

    all_train_neg_original_feats = change_format_for_feats(neg_feats_training)
    all_train_pos_original_feats = change_format_for_feats(pos_feats_training)
    all_neg_feats_testing = change_format_for_feats(neg_feats_testing)
    all_pos_feats_testing = change_format_for_feats(pos_feats_testing)

    #for model and loss initilization

    model = Linear_projection_MAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    # loss2 = nn.MSELoss()

    for epoch in range(1, args.epoch + 1):
        print("Epoch #", epoch)
        loss_epoch = model_train(train_feat, train_label, model, loss_fn, optimizer)
        val_acc, val_auc = model_test(model, val_feat, labels_test)
        orig_acc, orig_auc = model_test(model, all_test_original_feats, all_test_original_label)

        writer.add_scalar('Loss_training', loss_epoch, epoch)
        writer.add_scalar('Acc_extreme_samples_testset', val_acc, epoch)
        writer.add_scalar('AUC_extreme_samples_testset', val_auc, epoch)
        writer.add_scalar('Acc_orig_all_feats_directly_using_pretrained_classifier', orig_acc, epoch)
        writer.add_scalar('AUC_orig_all_feats_directly_using_pretrained_classifier', orig_auc, epoch)

        #Each time, the mapping is done on top of the original feats,
        # not on top of the new ones, so each time it is the original feats to new_feats

        new_train_neg_original_feats = model.projection_head(all_train_neg_original_feats).detach().numpy()
        new_train_pos_original_feats = model.projection_head(all_train_pos_original_feats).detach().numpy()

        # In the process of dynamic iteration, the only things that change are the samples and models picked each time,
        # and the mapping of each model is performed for the original MAE feature space, so when picking samples
        # to calculate the distance, the features that are mapped by the new model should be input each time.
        #
        #The purpose is to determine which the samples are and also their features should be their original MAE features,
        # so it is also necessary to go back and find the features corresponding to these samples.
        new_neg_feats_testing = model.projection_head(all_neg_feats_testing).detach().numpy()
        new_pos_feats_testing = model.projection_head(all_pos_feats_testing).detach().numpy()

        aucscore = get_score_and_dis_MAE(args.num_cluster,args.seed,new_train_neg_original_feats,new_neg_feats_testing,
                                     new_pos_feats_testing)

        writer.add_scalar('AUC_orig_all_feats_using_new_feats_and_ood_based', aucscore, epoch)

        print(aucscore)

        # state = {
        #     'epoch': epoch,
        #     'model': model.state_dict(),
        # }
        # torch.save(state, args.model_save_dir + str(epoch) + '.pth')

        dis_neg_train, dis_pos_train = get_score_and_dis_feats(args.num_cluster,args.seed,new_train_neg_original_feats,new_train_pos_original_feats)

        fintuning_feats, label = main_generator_dynamic(dis_neg_train, neg_feats_training, dis_pos_train, pos_feats_training)
        features_train, features_test, labels_train, labels_test = train_test_split(fintuning_feats, label,
                                                                                    random_state=12345,test_size=0.1,
                                                                                    shuffle=True)
        train_feat = change_format_for_feats(features_train)
        train_label = change_format_for_labels(labels_train)
        val_feat = change_format_for_feats(features_test)
        val_label = change_format_for_labels(labels_test)

        #for slide-level testing AUC

        testing_feats_original = np.load(args.testing_feats_original)
        testing_feats_original = change_format_for_feats(testing_feats_original)
        new_testing_feats_ = model.projection_head(testing_feats_original).detach().numpy()
        num_bag_list_index = np.load(args.num_bag_list_index)
        test_slide_label = np.load(args.test_slide_label)

        dis_new_testing_feats, dis_new_testing_feats_ = get_score_and_dis_feats(args.num_cluster, args.seed,
                                                               new_testing_feats_,
                                                               new_testing_feats_)
        slide_score_all = []
        slide_label_all = []
        for i in range(len(test_slide_label)):
            if i < len(test_slide_label)-1:
                start = num_bag_list_index[i]
                end = num_bag_list_index[i+1]
            if i == len(test_slide_label)-1:
                start = num_bag_list_index[i]
                end = len(dis_new_testing_feats)
            slide_score = np.mean(dis_new_testing_feats[start:end])
            slide_score_all.append(slide_score)
            if 'p' in test_slide_label[i]:
                slide_label_all.append(1)
            else:
                slide_label_all.append(0)

        slide_score_all = np.array(slide_score_all)
        slide_label_all = np.array(slide_label_all)
        slide_auc = get_auc_score(slide_label_all,slide_score_all)

        print(slide_auc)
        writer.add_scalar('Slide_AUC_using_new_feats_and_ood_based', slide_auc, epoch)

        print("")


    print("Finish!")



