import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models.neuralhash import NeuralHash
from adv3_robustness_check import get_dataset, get_hashes
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageNet
from tqdm import tqdm

from utils.hashing import load_hash_matrix
from utils.metrics import Accuracy
from utils.training import EarlyStopper


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_nn(model, optimizer, train_set, val_set, test_set=None, batch_size=32, num_workers=16, max_epochs=100):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = None
    if test_set is not None:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    early_stopper = EarlyStopper(window=args.early_stopping_window, min_diff=args.early_stopping_min_diff)
    running_val_loss = np.inf
    acc = Accuracy()
    epoch = 0
    best_val_acc = 0
    best_test_acc = 0
    while (not early_stopper.stop_early(running_val_loss)) and epoch < max_epochs:
        acc.reset()
        running_train_loss = 0
        model.train()
        for x, y in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            running_train_loss += loss.item() * len(x)
            optimizer.step()

            acc.update(output.softmax(1), y)
        train_acc = acc.compute_metric()
        running_train_loss /= len(train_set)

        model.eval()
        acc.reset()
        running_val_loss = 0
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            output = model(x)
            running_val_loss += criterion(output, y).item() * len(x)
            acc.update(output.softmax(1), y)
        val_acc = acc.compute_metric()
        running_val_loss /= len(val_set)

        test_acc = 0
        running_test_loss = 0
        if test_loader is not None:
            acc.reset()
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                output = model(x)
                running_test_loss += criterion(output, y).item() * len(x)
                acc.update(output.softmax(1), y)
            test_acc = acc.compute_metric()
            running_test_loss /= len(test_set)

        epoch += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(
            f'Epoch {epoch}: Train Acc: {train_acc:5.4f}, Train Loss: {running_train_loss:5.4f}, ' +
            f'Val Acc: {val_acc:5.4f} ({best_val_acc:5.4f}), Val Loss: {running_val_loss:5.4f}, ' +
            f'Test Acc: {test_acc:5.4f} ({best_test_acc:5.4f}), Test Loss: {running_test_loss:5.4f}'
        )


def use_imagenet_categories(dataset, hash_df, modify_dataset=False):
    # fix the duplicate 'crane'-label, 'maillot'-label, 'polecat'-label and 'missile'-label
    dataset.class_to_idx['crane (bird)'] = 134
    dataset.class_to_idx['maillot'] = 638
    dataset.class_to_idx['tank suit'] = 639
    dataset.class_to_idx['missile'] = 657
    dataset.class_to_idx['projectile'] = 744
    dataset.class_to_idx['polecat'] = 358
    imagenet_categories = pd.read_csv('imagenet_categories_modified.csv')

    # the new 'class_to_idx' dict
    category_to_idx = {}
    # the new 'classes' list
    categories = []
    # the new 'targets' list
    new_targets = np.full(hash_df.shape[0], -1)
    for index, row in imagenet_categories.iterrows():
        rows = row.values.tolist()
        category = rows.pop(0)
        category_to_idx[category] = index
        categories.append(category)

        classes = [x.replace("_", " ") for x in rows if str(x) != 'nan']
        for class_name in classes:
            # update the new target array
            target_indices = np.where(hash_df['target'].values == dataset.class_to_idx[class_name])[0]
            new_targets[target_indices] = index

    if modify_dataset:
        dataset.class_to_idx = category_to_idx
        dataset.classes = categories
        dataset.targets = new_targets.tolist()

    hash_df['target'] = new_targets.tolist()


def balance_classes(df):
    num_examples_per_class = np.array(
        [(df['target'].values == i).sum() for i in np.unique(df['target'].values)])
    min_num_examples = num_examples_per_class.min()

    indices = []
    for i in np.unique(df['target'].values):
        example_idx = np.where(df['target'].values == i)[0]
        if len(example_idx) > min_num_examples:
            selected_samples_idx = np.random.choice(example_idx, min_num_examples, replace=False)
            indices.append(selected_samples_idx)
        else:
            indices.append(example_idx)

    return indices


if __name__ == '__main__':
    TRAIN_DATASET = 'imagenet_train'
    VAL_DATASET = 'imagenet_val'
    NUM_CLASSES = 1000

    TRAIN_HASH_PATH = f'dataset_hashes/{TRAIN_DATASET}/{TRAIN_DATASET}_original_with_targets.csv'
    VAL_HASH_PATH = f'dataset_hashes/{VAL_DATASET}/{VAL_DATASET}_original_with_targets.csv'

    parser = argparse.ArgumentParser()

    parser.add_argument('--max_num_epochs', default=100, type=int, help='The maximum number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='The batch size used for training')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='The number of workers used for loading the training data')
    parser.add_argument('--lr', default=0.001, type=float, help='The learning rate used for learning')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str,
                        help='The optimizer used for training')
    parser.add_argument('--p_dropout', default=0.3, type=float, help='The dropout probability used during training')
    parser.add_argument('--layers', default=f'96,2048,4096,2048,{NUM_CLASSES}', type=str,
                        help='The number of neurons for the layers')
    parser.add_argument('--normalize_data', action='store_true', help='Whether to normalize the data to mean of 0 and unit variance')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='The weight decay used for training')
    parser.add_argument('--save_model', action='store_true', help='Whether to store the trained model')
    parser.add_argument('--use_imagenet_categories', action='store_true', help='Whether to use broader categories (there are 66 categories) for imagenet classification')
    parser.add_argument('--val_fraction', default=0.1, type=float, help='The fraction of the data that is used for evaluation')
    parser.add_argument('--early_stopping_window', default=10, type=int, help='The number of epochs for which there must be improvement')
    parser.add_argument('--early_stopping_min_diff', default=1e-4, type=float, help='The amount for which the metric should at least improve')
    parser.add_argument('--balance_classes', action='store_true', help='Whether to balance the number of samples per class of the training dataset')
    parser.add_argument('--disable_stratification', action='store_true', help='Whether to disable stratification when splitting the validation set from the train set')
    parser.add_argument('--seed', default=42, type=int, help='The seed for reproducibility')
    parser.add_argument('--use_normalized_bit_values', action='store_true', help='Whether to use -1 and 1 bit values instead of 0 and 1')

    args = parser.parse_args()
    args.layers = [int(x.replace("'", "")) for x in args.layers.split(',')]

    if args.use_imagenet_categories:
        NUM_CLASSES = 85

    if not (os.path.exists(TRAIN_HASH_PATH) and os.path.exists(VAL_HASH_PATH)):
        device = torch.device('cuda')

        model = NeuralHash()
        model.load_state_dict(torch.load('./models/model.pth'))
        model.eval()
        model = model.to(device)
        seed = torch.tensor(load_hash_matrix())
        seed = seed.to(device)

    if not os.path.exists(TRAIN_HASH_PATH):
        train_dataset = get_dataset(TRAIN_DATASET)
        binary_hashes, hex_hashes = get_hashes(train_dataset, model, seed, device, batch_size=128,
                                               num_workers=16)

        hash_df = pd.DataFrame(columns=['image', 'hash_bin', 'hash_hex', 'target'])
        hash_df = hash_df.assign(
            image=list(np.array(train_dataset.imgs)[:, 0]),
            hash_bin=binary_hashes,
            hash_hex=hex_hashes,
            target=train_dataset.targets
        )

        if not os.path.exists(os.path.dirname(TRAIN_HASH_PATH)):
            os.makedirs(os.path.dirname(TRAIN_HASH_PATH), exist_ok=False)
        hash_df.to_csv(TRAIN_HASH_PATH)

    if not os.path.exists(VAL_HASH_PATH):
        val_dataset = get_dataset(VAL_DATASET)
        binary_hashes, hex_hashes = get_hashes(val_dataset, model, seed, device, batch_size=32,
                                               num_workers=8)

        hash_df = pd.DataFrame(columns=['image', 'hash_bin', 'hash_hex', 'target'])
        hash_df = hash_df.assign(
            image=list(np.array(val_dataset.imgs)[:, 0]),
            hash_bin=binary_hashes,
            hash_hex=hex_hashes,
            target=val_dataset.targets
        )

        if not os.path.exists(os.path.dirname(VAL_HASH_PATH)):
            os.makedirs(os.path.dirname(VAL_HASH_PATH), exist_ok=False)
        hash_df.to_csv(VAL_HASH_PATH)

    # set everything for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # get the training data
    train_df = pd.read_csv(TRAIN_HASH_PATH)
    # get the validation data
    test_df = pd.read_csv(VAL_HASH_PATH)

    if args.use_imagenet_categories:
        dataset = get_dataset(VAL_DATASET)
        use_imagenet_categories(dataset, train_df)
        use_imagenet_categories(dataset, test_df)

    # convert the hashes from bit strings to numpy arrays
    train_targets = train_df['target'].to_numpy()
    train_vecs = np.array([[int(bit) for bit in list(hash)] for hash in train_df['hash_bin'].to_numpy()])
    test_targets = test_df['target'].to_numpy()
    test_vecs = np.array([[int(bit) for bit in list(hash)] for hash in test_df['hash_bin'].to_numpy()])

    if args.use_normalized_bit_values:
        train_vecs = (train_vecs - 0.5) * 2
        test_vecs = (test_vecs - 0.5) * 2

    if args.balance_classes:
        train_indices = balance_classes(train_df)
        test_indices = balance_classes(test_df)
    else:
        train_indices = [np.where(train_targets == i)[0] for i in range(NUM_CLASSES)]
        test_indices = [np.where(test_targets == i)[0] for i in range(NUM_CLASSES)]

    train_indices = np.concatenate(train_indices).flatten()
    test_indices = np.concatenate(test_indices).flatten()

    train_indices, val_indices = train_test_split(train_indices, test_size=args.val_fraction, random_state=args.seed, stratify=train_targets[train_indices] if not args.disable_stratification else None)

    train_vecs_set = train_vecs[train_indices]
    train_targets_set = train_targets[train_indices]
    val_vecs_set = train_vecs[val_indices]
    val_targets_set = train_targets[val_indices]
    test_vecs_set = test_vecs[test_indices]
    test_targets_set = test_targets[test_indices]

    if args.normalize_data:
        train_vecs_set = (train_vecs_set - train_vecs_set.mean(0)) / train_vecs_set.std(0)
    train_set = torch.utils.data.TensorDataset(torch.tensor(train_vecs_set).float(), torch.tensor(train_targets_set))
    val_set = torch.utils.data.TensorDataset(torch.tensor(val_vecs_set).float(), torch.tensor(val_targets_set))
    test_set = torch.utils.data.TensorDataset(torch.tensor(test_vecs_set).float(), torch.tensor(test_targets_set))

    # create the classifier
    neuron_nums = args.layers
    layers = []
    for i in range(1, len(neuron_nums)):
        layers.extend([
            nn.Linear(neuron_nums[i - 1], neuron_nums[i]),
            nn.BatchNorm1d(neuron_nums[i]),
            nn.Dropout(p=args.p_dropout),
            nn.ReLU()
        ])

    nn_classifier = nn.Sequential(
        *layers
    )
    nn_classifier.train()
    nn_classifier = nn_classifier.cuda()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(nn_classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(nn_classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError("Given optimizer name is not known")

    train_nn(nn_classifier, optimizer, train_set, val_set, test_set=test_set, batch_size=args.batch_size,
            num_workers=args.num_workers, max_epochs=args.max_num_epochs)

    if args.save_model:
        filename = f'./{TRAIN_DATASET}_{"categories_" if args.use_imagenet_categories else ""}hash_classifier{"_stratified" if not args.disable_stratification else ""}{"_balanced" if args.balance_classes else ""}_seed{args.seed}.pt'
        torch.save(nn_classifier.state_dict(), filename)