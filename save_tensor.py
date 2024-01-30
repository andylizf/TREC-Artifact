import torch

def save_forward_info(inputCentroids, vector_index, buckets_count, layer):
    torch.save(inputCentroids, "forward_inputCentroids_{}.pt".format(layer))
    torch.save(vector_index, "forward_vectorIndex_{}.pt".format(layer))
    torch.save(buckets_count, "forward_bucketsCount_{}.pt".format(layer))


def load_forward_info(layer):
    inputCentroids = torch.load("forward_inputCentroids_{}.pt".format(layer))
    vector_index = torch.load("forward_vectorIndex_{}.pt".format(layer))
    buckets_count = torch.load("forward_bucketsCount_{}.pt".format(layer))
    return inputCentroids, vector_index, buckets_count