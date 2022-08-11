from os import path
import sys
import argparse
import numpy as np


class TaobaoTxtToNpz:

    def __init__(
            self,
            datatype,
            mode,
            ts_length=20,
            points_per_user=4,
            numpy_rand_seed=7,
            raw_path="",
            pro_data="",
            spa_fea_sizes="",
            num_pts=1,    # pts to train or test
    ):
        # save arguments
        if mode == "train":
            self.numpy_rand_seed = numpy_rand_seed
        else:
            self.numpy_rand_seed = numpy_rand_seed + 31
        self.mode = mode
        # save dataset parameters
        self.total = num_pts    # number of lines in txt to process
        self.ts_length = ts_length
        self.points_per_user = points_per_user    # pos and neg points per user
        self.spa_fea_sizes = spa_fea_sizes
        self.M = 200    # max history length

        # split the datafile into path and filename
        lstr = raw_path.split("/")
        self.d_path = "/".join(lstr[0:-1]) + "/"
        self.d_file = lstr[-1]

        # preprocess data if needed
        if path.exists(str(pro_data)):
            print("Reading pre-processed data=%s" % (str(pro_data)))
            file = str(pro_data)
        else:
            file = str(pro_data)
            levels = np.fromstring(self.spa_fea_sizes, dtype=int, sep="-")
            if datatype == "taobao":
                self.Unum = levels[0]    # 987994  num of users
                self.Inum = levels[1]    # 4162024 num of items
                self.Cnum = levels[2]    # 9439    num of categories
                print("Reading raw data=%s" % (str(raw_path)))
                if self.mode == "test":
                    self.build_taobao_test(
                        raw_path,
                        file,
                    )
                else:
                    self.build_taobao_train_or_val(
                        raw_path,
                        file,
                    )
            elif datatype == "synthetic":
                self.build_synthetic_train_or_val(file,)
        # load data
        with np.load(file) as data:
            self.X_cat = data["X_cat"]
            self.X_int = data["X_int"]
            self.y = data["y"]

    # common part between train/val and test generation
    # truncates (if needed) and shuffles data points
    def truncate_and_save(self, out_file, do_shuffle, t, users, items, cats, times, y):
        # truncate. If for some users we didn't generate had too short history
        # we truncate the unused portion of the pre-allocated matrix.
        if t < self.total_out:
            users = users[:t, :]
            items = items[:t, :]
            cats = cats[:t, :]
            times = times[:t, :]
            y = y[:t]

        # shuffle
        if do_shuffle:
            indices = np.arange(len(y))
            indices = np.random.permutation(indices)
            users = users[indices]
            items = items[indices]
            cats = cats[indices]
            times = times[indices]
            y = y[indices]

        N = len(y)
        X_cat = np.zeros((3, N, self.ts_length + 1), dtype="i4")    # 4 byte int
        X_int = np.zeros((1, N, self.ts_length + 1), dtype=np.float)
        X_cat[0, :, :] = users
        X_cat[1, :, :] = items
        X_cat[2, :, :] = cats
        X_int[0, :, :] = times

        # saving to compressed numpy file
        if not path.exists(out_file):
            np.savez_compressed(
                out_file,
                X_cat=X_cat,
                X_int=X_int,
                y=y,
            )
        return

    # processes raw train or validation into npz format required by training
    # for train data out of each line in raw datafile produces several randomly chosen
    # datapoints, max number of datapoints per user is specified by points_per_user
    # argument, for validation data produces one datapoint per user.
    def build_taobao_train_or_val(self, raw_path, out_file):
        with open(str(raw_path)) as f:
            for i, _ in enumerate(f):
                if i % 50000 == 0:
                    print("pre-processing line: ", i)
        self.total = min(self.total, i + 1)

        print("total lines: ", self.total)

        self.total_out = self.total * self.points_per_user * 2    # pos + neg points
        print("Total number of points in raw datafile: ", self.total)
        print("Total number of points in output will be at most: ", self.total_out)
        np.random.seed(self.numpy_rand_seed)
        r_target = np.arange(0, self.M - 1)

        time = np.arange(self.ts_length + 1, dtype=np.int32) / (self.ts_length + 1)
        # time = np.ones(self.ts_length + 1, dtype=np.int32)

        users = np.zeros((self.total_out, self.ts_length + 1), dtype="i4")    # 4 byte int
        items = np.zeros((self.total_out, self.ts_length + 1), dtype="i4")    # 4 byte int
        cats = np.zeros((self.total_out, self.ts_length + 1), dtype="i4")    # 4 byte int
        times = np.zeros((self.total_out, self.ts_length + 1), dtype=np.float)
        y = np.zeros(self.total_out, dtype="i4")    # 4 byte int

        # determine how many datapoints to take from each user based on the length of
        # user behavior sequence
        # ind=0, 1, 2, 3,... t < 10, 20, 30, 40, 50, 60, ...
        k = 20
        regime = np.zeros(k, dtype=np.int)
        regime[1], regime[2], regime[3] = 1, 3, 6
        for j in range(4, k):
            regime[j] = self.points_per_user
        if self.mode == "val":
            self.points_per_user = 1
            for j in range(k):
                regime[j] = np.min([regime[j], self.points_per_user])
        last = self.M - 1    # max index of last item

        # try to generate the desired number of points (time series) per each user.
        # if history is short it may not succeed to generate sufficiently different
        # time series for a particular user.
        t, t_pos, t_neg, t_short = 0, 0, 0, 0
        with open(str(raw_path)) as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    print("processing line: ", i, t, t_pos, t_neg, t_short)
                if i >= self.total:
                    break
                units = line.strip().split("\t")
                item_hist_list = units[4].split(",")
                cate_hist_list = units[5].split(",")
                neg_item_hist_list = units[6].split(",")
                neg_cate_hist_list = units[7].split(",")
                user = np.array(np.maximum(np.int32(units[0]) - self.Inum, 0), dtype=np.int32)
                # y[i] = np.int32(units[3])
                items_ = np.array(list(map(lambda x: np.maximum(np.int32(x), 0), item_hist_list)), dtype=np.int32)
                cats_ = np.array(list(map(lambda x: np.maximum(np.int32(x) - self.Inum - self.Unum, 0),
                                          cate_hist_list)),
                                 dtype=np.int32)
                neg_items_ = np.array(list(map(lambda x: np.maximum(np.int32(x), 0), neg_item_hist_list)),
                                      dtype=np.int32)
                neg_cats_ = np.array(list(
                    map(lambda x: np.maximum(np.int32(x) - self.Inum - self.Unum, 0), neg_cate_hist_list)),
                                     dtype=np.int32)

                # select datapoints
                first = np.argmax(items_ > 0)
                ind = int((last - first) // 10)    # index into regime array
                # pos
                for _ in range(regime[ind]):
                    a1 = min(first + self.ts_length, last - 1)
                    end = np.random.randint(a1, last)
                    indices = np.arange(end - self.ts_length, end + 1)
                    if items_[indices[0]] == 0:
                        t_short += 1
                    items[t] = items_[indices]
                    cats[t] = cats_[indices]
                    users[t] = np.full(self.ts_length + 1, user)
                    times[t] = time
                    y[t] = 1
                    # check
                    if np.any(users[t] < 0) or np.any(items[t] < 0) \
                            or np.any(cats[t] < 0):
                        sys.exit("Categorical feature less than zero after \
                            processing. Aborting...")
                    t += 1
                    t_pos += 1
                # neg
                for _ in range(regime[ind]):
                    a1 = min(first + self.ts_length - 1, last - 1)
                    end = np.random.randint(a1, last)
                    indices = np.arange(end - self.ts_length + 1, end + 1)
                    if items_[indices[0]] == 0:
                        t_short += 1
                    items[t, :-1] = items_[indices]
                    cats[t, :-1] = cats_[indices]
                    neg_indices = np.random.choice(r_target, 1, replace=False)    # random final item
                    items[t, -1] = neg_items_[neg_indices]
                    cats[t, -1] = neg_cats_[neg_indices]
                    users[t] = np.full(self.ts_length + 1, user)
                    times[t] = time
                    y[t] = 0
                    # check
                    if np.any(users[t] < 0) or np.any(items[t] < 0) \
                            or np.any(cats[t] < 0):
                        sys.exit("Categorical feature less than zero after \
                        processing. Aborting...")
                    t += 1
                    t_neg += 1

        print("total points, pos points, neg points: ", t, t_pos, t_neg)

        self.truncate_and_save(out_file, True, t, users, items, cats, times, y)
        return

    # processes raw test datafile into npz format required to be used by
    # inference step, produces one datapoint per user by taking last ts-length items
    def build_taobao_test(self, raw_path, out_file):

        with open(str(raw_path)) as f:
            for i, _ in enumerate(f):
                if i % 50000 == 0:
                    print("pre-processing line: ", i)
        self.total = i + 1

        self.total_out = self.total    # pos + neg points
        print("ts_length: ", self.ts_length)
        print("Total number of points in raw datafile: ", self.total)
        print("Total number of points in output will be at most: ", self.total_out)

        time = np.arange(self.ts_length + 1, dtype=np.int32) / (self.ts_length + 1)

        users = np.zeros((self.total_out, self.ts_length + 1), dtypei4="")    # 4 byte int
        items = np.zeros((self.total_out, self.ts_length + 1), dtype="i4")    # 4 byte int
        cats = np.zeros((self.total_out, self.ts_length + 1), dtype="i4")    # 4 byte int
        times = np.zeros((self.total_out, self.ts_length + 1), dtype=np.float)
        y = np.zeros(self.total_out, dtype="i4")    # 4 byte int

        # try to generate the desired number of points (time series) per each user.
        # if history is short it may not succeed to generate sufficiently different
        # time series for a particular user.
        t, t_pos, t_neg = 0, 0, 0
        with open(str(raw_path)) as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    print("processing line: ", i, t, t_pos, t_neg)
                if i >= self.total:
                    break
                units = line.strip().split("\t")
                item_hist_list = units[4].split(",")
                cate_hist_list = units[5].split(",")

                user = np.array(np.maximum(np.int32(units[0]) - self.Inum, 0), dtype=np.int32)
                y[t] = np.int32(units[3])
                items_ = np.array(list(map(lambda x: np.maximum(np.int32(x), 0), item_hist_list)), dtype=np.int32)
                cats_ = np.array(list(map(lambda x: np.maximum(np.int32(x) - self.Inum - self.Unum, 0),
                                          cate_hist_list)),
                                 dtype=np.int32)

                # get pts
                items[t] = items_[-(self.ts_length + 1):]
                cats[t] = cats_[-(self.ts_length + 1):]
                users[t] = np.full(self.ts_length + 1, user)
                times[t] = time
                # check
                if np.any(users[t] < 0) or np.any(items[t] < 0) \
                        or np.any(cats[t] < 0):
                    sys.exit("Categorical feature less than zero after \
                        processing. Aborting...")
                if y[t] == 1:
                    t_pos += 1
                else:
                    t_neg += 1
                t += 1

        print("total points, pos points, neg points: ", t, t_pos, t_neg)

        self.truncate_and_save(out_file, False, t, users, items, cats, times, y)
        return

    # builds small synthetic data mimicking the structure of taobao data
    def build_synthetic_train_or_val(self, out_file):

        np.random.seed(123)
        fea_sizes = np.fromstring(self.spa_fea_sizes, dtype=int, sep="-")
        maxval = np.min(fea_sizes)
        num_s = len(fea_sizes)
        X_cat = np.random.randint(maxval, size=(num_s, self.total, self.ts_length + 1), dtype="i4")    # 4 byte int
        X_int = np.random.uniform(0, 1, size=(1, self.total, self.ts_length + 1))
        y = np.random.randint(0, 2, self.total, dtype="i4")    # 4 byte int

        # saving to compressed numpy file
        if not path.exists(out_file):
            np.savez_compressed(
                out_file,
                X_cat=X_cat,
                X_int=X_int,
                y=y,
            )
        return


# creates a loader (train, val or test data) to be used in the main training loop
# or during inference step
def make_tbsm_data_and_loader(args, mode):
    if mode == "train":
        raw = args.raw_train_file
        proc = args.pro_train_file
        numpts = args.num_train_pts
    elif mode == "val":
        raw = args.raw_train_file
        proc = args.pro_val_file
        numpts = args.num_val_pts
    else:
        raw = args.raw_test_file
        proc = args.pro_test_file
        numpts = 1

    TaobaoTxtToNpz(
        args.datatype,
        mode,
        args.ts_length,
        args.points_per_user,
        args.numpy_rand_seed,
        raw,
        proc,
        args.arch_embedding_size,
        numpts,
    )


def main(args):
    make_tbsm_data_and_loader(args, 'train')
    make_tbsm_data_and_loader(args, 'val')
    make_tbsm_data_and_loader(args, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datatype", type=str, default="taobao")
    parser.add_argument("--raw-train-file", type=str, default="./input/train.txt")
    parser.add_argument("--pro-train-file", type=str, default="./output/train.npz")
    parser.add_argument("--raw-test-file", type=str, default="./input/test.txt")
    parser.add_argument("--pro-test-file", type=str, default="./output/test.npz")
    parser.add_argument("--pro-val-file", type=str, default="./output/val.npz")
    parser.add_argument("--ts-length", type=int, default=20)
    parser.add_argument("--num-train-pts", type=int, default=100)
    parser.add_argument("--num-val-pts", type=int, default=20)
    parser.add_argument("--points-per-user", type=int, default=10)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")    # vectors
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    args = parser.parse_args()
    main(args)
