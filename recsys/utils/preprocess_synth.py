import torch
# a = torch.tensor([1, 3, 5, 7, 14, 3, 40, 23])
# b, c = torch.unique(a, sorted=True, return_inverse=True)
# print(b)
# print(c)

BATCH_SIZE = 65536
TABLLE_NUM = 856
FILE_LIST = [f"/data/scratch/RecSys/embedding_bag/fbgemm_t856_bs65536_{i}.pt" for i in range(
    16)] + ["/data/scratch/RecSys/embedding_bag/fbgemm_t856_bs65536.pt"]
KEYS = []
for i in range(TABLLE_NUM):
    KEYS.append("table_{}".format(i))

CHOSEN_TABLES = [i for i in range(0, 856)]

def load_file(file_path, cuda=True):
    indices, offsets, lengths = torch.load(file_path)
    if cuda:
        indices = indices.int().cuda()
        offsets = offsets.int().cuda()
        lengths = lengths.int().cuda()
    else :
        indices = indices.int()
        offsets = offsets.int()
        lengths = lengths.int()
    indices_per_table = []
    for i in range(TABLLE_NUM):
        if i not in CHOSEN_TABLES:
            continue
        start_pos = offsets[i * BATCH_SIZE]
        end_pos = offsets[i * BATCH_SIZE + BATCH_SIZE]
        part = indices[start_pos:end_pos]
        indices_per_table.append(part)
    return indices_per_table

if __name__ == "__main__":
    indices_per_table_list= []
    indices_per_table_length_list = []
    for i, file in enumerate(FILE_LIST):
        indices_per_table = load_file(file, cuda=False)
        print("loaded ", file)
        for j, indices in enumerate(indices_per_table):
            if i == 0:
                indices_per_table_list.append([indices])
                indices_per_table_length_list.append([indices.shape[0]])
            else:
                indices_per_table_list[j].append(indices)
                indices_per_table_length_list[j].append(indices.shape[0])
                
    # unique op for each table:
    for i, (indices_list, length_list) in enumerate(zip(indices_per_table_list, indices_per_table_length_list)):
        catted = torch.cat(indices_list)
        _, processed = torch.unique(catted, sorted=True, return_inverse=True)
        indices_per_table_list[i] = torch.split(processed, length_list)
        
    # save to each file:
    for i, file in enumerate(FILE_LIST):
        _, offsets, lengths = torch.load(file)
        indices_per_table = [indices_in_table[i] for indices_in_table in indices_per_table_list]
        reconcatenate = torch.cat(indices_per_table)
        torch.save((reconcatenate, offsets, lengths),
                   f"/home/lccsr/data2/embedding_bag_processed/fbgemm_t856_bs65536_processed_{i}.pt")
        print("saved, ", i)
