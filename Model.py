from torch.utils.data import Dataset, DataLoader
from scipy.sparse import load_npz

#defining the class Vectorized_Dataset
class Vectorized_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#loading the vectorized datasets into the script
path = "C:/Users/dernj/Desktop/vectorized_dataset"

#binary Countvectorize
bin_test = "/data_bin_test.npz"
bin_train = "/data_bin_train.npz"
bin_val = "/data_bin_val.npz"
data_bin_test = load_npz(path + bin_test)
data_bin_train = load_npz(path + bin_train)
data_bin_val = load_npz(path + bin_val)

#non-binary Countvectorize
freq_test = "/data_freq_test.npz"
freq_train = "/data_freq_train.npz"
freq_val = "/data_freq_val.npz"
data_freq_test = load_npz(path + freq_test)
data_freq_train = load_npz(path + freq_train)
data_freq_val = load_npz(path + freq_val)

#hashtrick vectorizer
hash_test = "/data_hash_test.npz"
hash_train = "/data_hash_train.npz"
hash_val = "/data_hash_val.npz"
data_hash_test = load_npz(path + hash_test)
data_hash_train = load_npz(path + hash_train)
data_hash_val = load_npz(path + hash_val)

#tfidf vectorizer
tfidf_test = "/data_tfidf_test.npz"
tfidf_train = "/data_tfidf_train.npz"
tfidf_val = "/data_tfidf_val.npz"
data_tfidf_test = load_npz(path + tfidf_test)
data_tfidf_train = load_npz(path + tfidf_train)
data_tfidf_val = load_npz(path + tfidf_val)