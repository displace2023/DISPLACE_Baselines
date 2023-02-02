import os
import matplotlib.pyplot as plt
import fastcluster
import scipy.cluster.hierarchy as sch
import argparse
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import numpy as np
import scipy as sp
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import RandomizedSearchCV
from pathlib import Path
from pdb import set_trace as bp
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.special import softmax
from scipy.linalg import eigh


def twoGMMcalib_lin(s, niters=10):
    """
    Train two-Gaussian GMM with shared variance for calibration of scores 's'
    Returns threshold for original scores 's' that "separates" the two gaussians
    and array of linearly callibrated log odds ratio scores.
    """
    weights = np.array([0.5, 0.5])
    means = np.mean(s) + np.std(s) * np.array([-1, 1])
    var = np.var(s)
    threshold = np.inf
    for _ in range(niters):
        lls = np.log(weights) - 0.5 * np.log(var) - 0.5 * (s[:, np.newaxis] - means)**2 / var
        gammas = softmax(lls, axis=1)
        cnts = np.sum(gammas, axis=0)
        weights = cnts / cnts.sum()
        means = s.dot(gammas) / cnts
        var = ((s**2).dot(gammas) / cnts - means**2).dot(weights)
        threshold = -0.5 * (np.log(weights**2 / var) - means**2 / var).dot([1, -1]) / (means/var).dot([1, -1])
    return threshold, lls[:, means.argmax()] - lls[:, means.argmin()]


def cos_similarity(x):
    """Compute cosine similarity matrix in CPU & memory sensitive way

    Args:
        x (np.ndarray): embeddings, 2D array, embeddings are in rows

    Returns:
        np.ndarray: cosine similarity matrix

    """

    assert x.ndim == 2, f'x has {x.ndim} dimensions, it must be matrix'
    x = x / (np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)) + 1.0e-32)
    # return x.dot(x.T)
    assert np.allclose(np.ones_like(x[:, 0]), np.sum(np.square(x), axis=1))
    max_n_elm = 200000000
    step = max(max_n_elm // (x.shape[0] * x.shape[0]), 1)
    retval = np.zeros(shape=(x.shape[0], x.shape[0]), dtype=np.float64)
    x0 = np.expand_dims(x, 0)
    x1 = np.expand_dims(x, 1)
    for i in range(0, x.shape[1], step):
        product = x0[:, :, i:i+step] * x1[:, :, i:i+step]
        retval += np.sum(product, axis=2, keepdims=False)
    assert np.all(retval >= -1.0001), retval
    assert np.all(retval <= 1.0001), retval
    return retval


def AHC(x):
	scr_mx = cos_similarity(x)
	thr, _ = twoGMMcalib_lin(scr_mx.ravel())
	scr_mx = squareform(-scr_mx, checks=False)
	lin_mat = fastcluster.linkage(scr_mx, method='average', preserve_input='False')
	del scr_mx
	adjust = abs(lin_mat[:, 2].min())
	lin_mat[:, 2] += adjust
	labels1st = fcluster(lin_mat, -(thr - 0.015) + adjust,criterion='distance') - 1
	return labels1st


def labels_to_rttm(segments, labels, rttm_file, rttm_channel=1):
    labels = labels+1
    reco2segs = {}


    with open(segments, 'r') as segments_file:
        lines = segments_file.readlines()
    for line, label in zip(lines, labels):
        seg, reco, start, end = line.strip().split()
        start, end = float(start), float(end)
        
        try:
            if reco in reco2segs:
                reco2segs[reco] = "{} {},{},{}".format(reco2segs[reco],start,end,label)  #reco2segs[reco] + " " + start + "," + end + "," + label
            else:
                reco2segs[reco] = "{} {},{},{}".format(reco,start,end,label) #reco + " " + start + "," + end + "," + label
        except KeyError:
            raise RuntimeError("Missing label for segment {0}".format(seg))
        
    contiguous_segs = []
    for reco in sorted(reco2segs):
        segs = reco2segs[reco].strip().split()
        new_segs = ""
        for i in range(1, len(segs)-1):
            start, end, label = segs[i].split(',')
            next_start, next_end, next_label = segs[i+1].split(',')
            if float(end) > float(next_start):
                done = False
                avg = str((float(next_start) + float(end)) / 2.0)
                segs[i+1] = ','.join([avg, next_end, next_label])
                new_segs += " {},{},{}".format(start,avg,label)   #" " + start + "," + avg + "," + label
            else:
                new_segs += " {},{},{}".format(start,end,label)   #" " + start + "," + end + "," + label
        start, end, label = segs[-1].split(',')
        new_segs += " {},{},{}".format(start,end,label)  #" " + start + "," + end + "," + label
        contiguous_segs.append(reco + new_segs)
        
    merged_segs = []
    for reco_line in contiguous_segs:
        segs = reco_line.strip().split()
        reco = segs[0]
        new_segs = ""
        for i in range(1, len(segs)-1):
            start, end, label = segs[i].split(',')
            next_start, next_end, next_label = segs[i+1].split(',')
            if float(end) == float(next_start) and label == next_label:
                segs[i+1] = ','.join([start, next_end, next_label])
            else:
                new_segs += " {},{},{}".format(start,end,label)  #" " + start + "," + end + "," + label
        start, end, label = segs[-1].split(',')
        new_segs += " {},{},{}".format(start,end,label)  #" " + start + "," + end + "," + label
        merged_segs.append(reco + new_segs)
        
    with open(rttm_file, 'w') as rttm_writer:
        for reco_line in merged_segs:
            segs = reco_line.strip().split()
            reco = segs[0]
            for i in range(1, len(segs)):
                start, end, label = segs[i].strip().split(',')
                print("LANGUAGE {0} {1} {2:7.3f} {3:7.3f} <NA> <NA> L{4} <NA> <NA>".format(
                    reco, rttm_channel, float(start), float(end)-float(start), label), file=rttm_writer)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings_segments_list") # emb_segments.list or llk_segments.list
    parser.add_argument("rttm_path") # exps/exp1/ecapa_tdnn_voxlingua_speechbrain_language_embeddings/rttm_outputs
    parser.add_argument('mode', type=str, help='AHC,kmeans,spectral_clustering')
    args = parser.parse_args()
    embeddings_segments = np.genfromtxt(args.embeddings_segments_list, dtype=str)
    return embeddings_segments, args.rttm_path, args.mode

def out_filename(emb_file, rttm_path, clustering_hparams=""):
    rec_basename = os.path.splitext(os.path.basename(emb_file))[0]
    segment_basename = os.path.basename(os.path.dirname(emb_file))
    # bp()
    out_file = os.path.join(rttm_path, segment_basename, f"{clustering_hparams}", f"{rec_basename}.rttm")
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    return out_file

    
def main():
    embeddings_segments, rttm_path, mode = get_args()
    
    clustering_algo_dict = {"AHC": AHC}
    
    cluster = clustering_algo_dict[mode]
    
    
    for emb_file, segments in embeddings_segments:
        data = np.load(emb_file)
	
        labels = cluster(data)
        out_file = out_filename(emb_file, rttm_path, mode)
        labels_to_rttm(segments, labels, out_file)
        print(f"\n\n\n\n Clustering ({mode}) done for {emb_file}, saved to {out_file}.\n\n\n\n")
        
        
main()
