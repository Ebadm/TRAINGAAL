from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch
from GANModels import *
from swat_loader import *
from torch.autograd.variable import Variable
import numpy as np
import os
from visualizationMetrics import visualization
import numpy as np
from scipy.special import kl_div
import matplotlib.pyplot as plt
import gc
from scipy import stats
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score


def calculate_mse(array1, array2):
    assert array1.shape == array2.shape, "Arrays must have the same shape"
    squared_diff = np.square(array1 - array2)
    mse = np.mean(squared_diff)
    return mse


class Synthetic_Dataset(Dataset):
    def __init__(self):
        gen_net = Generator(seq_len=60).to(device)
        generator = torch.load('generator.pth', map_location=device)
        gen_net.load_state_dict(generator)
        dis_net = Discriminator(seq_len=60).to(device)
        discriminator = torch.load('discriminator.pth', map_location=device)
        gen_net.load_state_dict(generator)
        dis_net.load_state_dict(discriminator)
        dataset = swat_load_dataset(is_train=True, is_attack=True)
        labels = dataset.get_labels_attack()
        print(np.unique(labels))
        unique, counts = np.unique(labels, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        print(counts_dict)

        data_loader = data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        anomaly_scores = []
        g_scores = []
        d_scores = []
        anomaly = []
        total, hit = 0, 0
        tp, fp, fn = 0, 0, 0

        with torch.no_grad():
            for batch_idx, (real_data, labels) in enumerate(data_loader):
                real_data = real_data.to(device)
                real_data = Variable(real_data).to(device)
                batch_size, sliding_window_length, features_size = real_data.size()
                real_data_avg = torch.mean(real_data, dim=1)
                real_data_avg = real_data_avg.view(batch_size, features_size)
                synthetic_sample = gen_net(real_data_avg)
                discriminator_output_real = dis_net(real_data)
                discriminator_output_syn = dis_net(synthetic_sample)
                d_loss = -torch.mean(discriminator_output_real) + torch.mean(discriminator_output_syn)
                g_loss = -torch.mean(discriminator_output_syn)
                alpha = 0.25
                ads_loss = (1 - alpha) * d_loss -  alpha * g_loss
                anomaly_scores.append(ads_loss.item())
                d_scores.append(d_loss.item())
                g_scores.append(g_loss.item())
                boundary_res = ads_loss.item() > 4
                anomaly.append(boundary_res)
                most_common_label = stats.mode(labels.numpy()).mode[0]
                hit = hit + 1 if most_common_label==boundary_res else hit
                total += 1
                if most_common_label and boundary_res:
                    tp += 1
                elif not most_common_label and boundary_res:
                    fp += 1
                elif most_common_label and not boundary_res:
                    fn += 1

                real_data = real_data.cpu().detach().numpy()
                synthetic_sample = synthetic_sample.cpu().detach().numpy()
                print("Batch [{}/{}], ADS: {:.4f} [GLOSS: {:.4f} | DLOSS: {:.4f} for {:.4f}]".format(batch_idx, len(data_loader), ads_loss.item(), g_loss, d_loss, most_common_label))


        anomalies = np.array(anomaly)
        anomaly_scores = np.array(anomaly_scores)
        print("AVERAGE: ", np.mean(anomaly_scores))
        print("MAX: ", np.max(anomaly_scores))
        print("MIN: ", np.min(anomaly_scores))
        print("AVG GLOSS: {:.4f} | MAX GLOSS: {:.4f} | MIN GLOSS: {:.4f}]".format(np.mean(g_scores), np.max(g_scores), np.min(g_scores)))
        print("AVG DLOSS: {:.4f} | MAX DLOSS: {:.4f} | MIN DLOSS: {:.4f}]".format(np.mean(d_scores), np.max(d_scores), np.min(d_scores)))
        print("Accuracy: ", hit/total)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1)


enable_cuda = True
# Check if CUDA is available and enabled
if enable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
    gc.collect()
    torch.cuda.empty_cache()
    print("CUDA is enabled. Training on GPU.")
    
else:
    device = torch.device('cpu')
    print("Training on CPU.")

synthetic_dataset = Synthetic_Dataset()
