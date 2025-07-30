import numpy as np
import pandas as pd
import os
import copy
import random
import logging
import datetime
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import models, datasets
import time
import gc  # For garbage collection

# Set up logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{log_dir}/fl_training_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger()

torch.set_num_threads(1)

# ==== Configuration ====
target_size = (224, 224)
batch_size = 64
n_rounds = 100
local_epochs = 5
NUM_CLIENTS = 100
SEED = 43
CLIENT_SPLIT = "dirichlet"
DIRICHLET_ALPHA = 0.3
PRIVACY_NOISE_LEVEL = 0.01
MAHALANOBIS_THRESHOLD = 150.0
TOP_K_CLIENTS = 30
NUM_WORKERS = 2
PIN_MEMORY = True
PREFETCH_FACTOR = 2
USE_MIXED_PRECISION = False

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==== MNIST Dataset & Transforms ====
mnist_mean = [0.1307, 0.1307, 0.1307]
mnist_std = [0.3081, 0.3081, 0.3081]

train_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std),
    ]
)


def load_mnist():
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform
    )
    return train_dataset, test_dataset


num_classes = 10


def get_client_data_splits(train_dataset, num_clients, split_type, alpha):
    indices = list(range(len(train_dataset)))
    labels = [train_dataset[i][1] for i in indices]
    client_indices = []

    if split_type == "dirichlet":
        label_indices = {}
        for idx, label in enumerate(labels):
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(idx)
        for class_id in range(num_classes):
            class_indices = label_indices[class_id]
            proportions = np.random.dirichlet([alpha] * num_clients)
            num_samples_per_client = (proportions * len(class_indices)).astype(int)
            remaining = len(class_indices) - sum(num_samples_per_client)
            if remaining > 0:
                indices_to_add = np.random.choice(num_clients, remaining, replace=False)
                for idx in indices_to_add:
                    num_samples_per_client[idx] += 1
            random.shuffle(class_indices)
            start_idx = 0
            for client_id in range(num_clients):
                if len(client_indices) <= client_id:
                    client_indices.append([])
                samples_for_client = num_samples_per_client[client_id]
                client_indices[client_id].extend(
                    class_indices[start_idx : start_idx + samples_for_client]
                )
                start_idx += samples_for_client

    client_datasets = []
    for indices in client_indices:
        client_datasets.append(Subset(train_dataset, indices))

    print("\n📊 Data distribution across clients:")
    client_label_counts = []
    for i, indices_subset in enumerate(client_indices):
        labels_subset = [labels[idx] for idx in indices_subset]
        label_counts = {
            label: labels_subset.count(label) for label in range(num_classes)
        }
        client_label_counts.append(label_counts)
    distribution_df = pd.DataFrame(client_label_counts)
    print(distribution_df)
    total_per_class = {
        label: sum(
            counts[label] if label in counts else 0 for counts in client_label_counts
        )
        for label in range(num_classes)
    }
    percent_dfs = []
    for client_id, counts in enumerate(client_label_counts):
        percents = {
            label: (
                (counts.get(label, 0) / total_per_class[label] * 100)
                if total_per_class[label] > 0
                else 0
            )
            for label in range(num_classes)
        }
        percent_dfs.append(percents)
    print("\n📊 Percentage distribution across clients:")
    percent_df = pd.DataFrame(percent_dfs)
    print(percent_df.round(2))
    return client_datasets


def get_resnet18_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_model(model, train_loader, epochs, device, client_id=None):
    start_time = time.time()
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    epoch_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / total if total > 0 else 0
        epoch_acc = 100 * correct / total if total > 0 else 0
        epoch_losses.append(avg_loss)
        logger.info(
            f"Client {client_id} Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {epoch_acc:.2f}%"
        )
    torch.cuda.empty_cache()
    gc.collect()
    train_time = time.time() - start_time
    if client_id is not None:
        logger.info(
            f"⏱️ Client {client_id} training completed in {train_time:.2f} seconds"
        )
    return train_time, epoch_losses


def evaluate_model(model, loader, device):
    start_time = time.time()
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0
    eval_time = time.time() - start_time
    logger.info(
        f"⏱️ Model evaluation completed in {eval_time:.2f} seconds with accuracy {accuracy:.2f}%"
    )
    return accuracy


def extract_features(model, dataloader, client_id=None):
    start_time = time.time()
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model.conv1(x)
            out = model.bn1(out)
            out = model.relu(out)
            out = model.maxpool(out)
            out = model.layer1(out)
            out = model.layer2(out)
            out = model.layer3(out)
            out = model.layer4(out)
            out = nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(x.size(0), -1)
            features.append(out.cpu().numpy())
            labels.append(y.numpy())
    feat_time = time.time() - start_time
    if client_id is not None:
        logger.info(
            f"⏱️ Feature extraction for Client {client_id} completed in {feat_time:.2f} seconds"
        )
    return np.vstack(features), np.concatenate(labels)


def mahalanobis_dist(x, mu, sigma_inv):
    delta = x - mu
    values = np.einsum("ij,jk,ik->i", delta, sigma_inv, delta)
    values = np.maximum(values, 0)
    return np.sqrt(values)


def client_mahalanobis_filtering(
    features, global_mu, global_sigma_inv, threshold=None, percentile=90.0
):
    dists = mahalanobis_dist(features, global_mu, global_sigma_inv)
    if threshold is None:
        adaptive_threshold = np.percentile(dists, percentile)
        keep_indices = np.where(dists < adaptive_threshold)[0]
    else:
        keep_indices = np.where(dists < threshold)[0]
    if len(keep_indices) < 0.2 * len(features):
        min_samples = max(int(0.2 * len(features)), 100)
        min_samples = min(min_samples, len(features))
        keep_indices = np.argsort(dists)[:min_samples]
    return keep_indices, features[keep_indices]


def compute_client_distribution(features, noise_level=PRIVACY_NOISE_LEVEL):
    mu = np.mean(features, axis=0)
    cov = np.cov(features.T) + np.eye(features.shape[1]) * 1e-6
    noisy_mu = mu + np.random.normal(0, noise_level, mu.shape)
    noisy_cov_raw = cov + np.random.normal(0, noise_level * 5, cov.shape)
    cov = ensure_psd(cov)
    noisy_cov = ensure_psd(noisy_cov_raw)
    return mu, cov, noisy_mu, noisy_cov


def ensure_psd(cov_matrix, epsilon=1e-6):
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    try:
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, epsilon)
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon
    except np.linalg.LinAlgError:
        n = cov_matrix.shape[0]
        cov_matrix = np.eye(n) * epsilon
    return cov_matrix


def initialize_random_global_distribution(feature_dim=512):
    random_mean = np.random.normal(0, 0.1, feature_dim)
    A = np.random.normal(0, 0.1, (feature_dim, feature_dim))
    random_cov = np.dot(A, A.T) + np.eye(feature_dim) * 0.1
    random_cov = ensure_psd(random_cov)
    return random_mean, random_cov


def weighted_fedavg(models, weights):
    num_models = len(models)
    if num_models == 0:
        return {}
    if num_models == 1:
        return copy.deepcopy(models[0].state_dict())
    state_dicts = [model.state_dict() for model in models]
    avg_state = {}
    for key in state_dicts[0]:
        if all(key in sd for sd in state_dicts):
            avg_state[key] = sum(w * sd[key] for w, sd in zip(weights, state_dicts))
        else:
            avg_state[key] = state_dicts[0][key].clone()
    return avg_state


def enhanced_fedavg_main(num_rounds, local_epochs):
    gta = []
    logger.info(
        f"🚀 Starting Federated Learning with {NUM_CLIENTS} clients on device: {device}"
    )
    logger.info(
        f"📊 Configuration: batch_size={batch_size}, rounds={num_rounds}, local_epochs={local_epochs}, alpha={DIRICHLET_ALPHA}"
    )
    logger.info(
        f"🛠️ Using Mahalanobis data filtering with RANDOM client selection and simple FedAvg"
    )

    train_dataset, test_dataset = load_mnist()
    logger.info(f"⏱️ Dataset loading completed.")

    client_datasets = get_client_data_splits(
        train_dataset, NUM_CLIENTS, CLIENT_SPLIT, DIRICHLET_ALPHA
    )
    logger.info(f"⏱️ Client data splitting completed.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
    )

    global_model = get_resnet18_model(num_classes=num_classes).to(device)
    global_weights = global_model.state_dict()

    global_mu, global_cov = initialize_random_global_distribution(feature_dim=512)
    global_sigma_inv = np.linalg.inv(global_cov + np.eye(512) * 1e-6)
    logger.info("🔰 Initialized random global distribution for Round 0")

    for round_idx in range(num_rounds):
        round_start = time.time()
        logger.info(f"\n--- 🔄 Global Round {round_idx+1}/{num_rounds} Started ---")

        client_models = []
        client_mus = []
        client_covs = []
        client_num_samples = []

        for i, client_dataset in enumerate(client_datasets):
            client_start = time.time()
            logger.info(f" Processing Client {i+1}/{NUM_CLIENTS}")
            local_model = get_resnet18_model(num_classes=num_classes).to(device)
            local_model.load_state_dict(global_weights)

            loader = DataLoader(
                client_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH_FACTOR,
            )
            features, labels = extract_features(local_model, loader, client_id=i + 1)

            orig_mu, orig_cov, noisy_mu, noisy_cov = compute_client_distribution(
                features
            )

            keep_indices, filtered_features = client_mahalanobis_filtering(
                features,
                global_mu,
                global_sigma_inv,
                threshold=None,
                percentile=90.0,
            )

            logger.info(
                f"   Client {i+1}: {len(keep_indices)}/{len(features)} samples kept after filtering"
            )

            if len(keep_indices) > 10:
                filtered_data = Subset(client_dataset, keep_indices)
                train_loader = DataLoader(
                    filtered_data,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                )
                train_time, losses = train_model(
                    local_model,
                    train_loader,
                    epochs=local_epochs,
                    device=device,
                    client_id=i + 1,
                )
                client_models.append(local_model)
                client_mus.append(noisy_mu)
                client_covs.append(noisy_cov)
                client_num_samples.append(len(keep_indices))
            else:
                logger.warning(
                    f"   ⚠️ Client {i+1} has too few samples after filtering, skipping."
                )

            client_time = time.time() - client_start
            logger.info(
                f"⏱️ Client {i+1} processing completed in {client_time:.2f} seconds"
            )

        if client_mus:
            global_mu = np.mean(client_mus, axis=0)
            global_cov = np.mean(client_covs, axis=0)
            global_sigma_inv = np.linalg.inv(
                global_cov + np.eye(global_cov.shape[0]) * 1e-6
            )
            logger.info("   Updated global distribution from client distributions")

        selected_models = []
        if len(client_models) > 0:
            if len(client_models) > TOP_K_CLIENTS:
                selected_indices = np.random.choice(
                    len(client_models), TOP_K_CLIENTS, replace=False
                ).tolist()
                selected_models = [client_models[i] for i in selected_indices]
                logger.info(
                    f"🔍 Randomly selected {TOP_K_CLIENTS} out of {len(client_models)} clients: {[i+1 for i in selected_indices]}"
                )
            else:
                selected_models = client_models
                logger.info(f"🔍 Using all {len(client_models)} available clients")
        else:
            logger.warning(
                "⚠️ No clients passed filtering in this round! "
                "Skipping aggregation and keeping previous global model."
            )

        logger.info("🔄 Step 4: Simple FedAvg Model Aggregation")
        if len(selected_models) > 0:
            equal_weights = [1.0 / len(selected_models)] * len(selected_models)
            logger.info(
                f"⚖️ Using equal weights for all selected clients: {round(equal_weights[0], 4)}"
            )
            global_weights = weighted_fedavg(selected_models, equal_weights)
            global_model.load_state_dict(global_weights)
            acc = evaluate_model(global_model, test_loader, device=device)
            logger.info(f"📈 Test Accuracy after Round {round_idx+1}: {acc:.2f}%")
            gta.append(acc)
            np.save("./Filtered_MNIST_RandomSelect_SimpleFedAvg.npy", gta)
        else:
            logger.warning(
                "⚠️ No client models available for aggregation in this round!"
            )

        round_time = time.time() - round_start
        logger.info(f"⏱️ Round {round_idx+1} completed in {round_time:.2f} seconds")
        del client_models
        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - round_start
    logger.info(f"✅ Federated Learning completed in {total_time:.2f} seconds")
    logger.info(f"📈 Final Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    overall_start_time = time.time()
    enhanced_fedavg_main(num_rounds=n_rounds, local_epochs=local_epochs)
    execution_time = time.time() - overall_start_time
    logger.info(
        f"✅ Total execution time: {execution_time:.2f} seconds ({execution_time/3600:.2f} hours)"
    )
