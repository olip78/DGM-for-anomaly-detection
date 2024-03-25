import torch
from torch import optim
import torch.distributions as TD

from collections import defaultdict
from tqdm import tqdm
from typing import Tuple


from matplotlib import pyplot as plt
from typing import Dict, Tuple



def get_normal_KL(mean_1, log_std_1, mean_2=None, log_std_2=None):
    """
        returns the value of KL(p1 || p2),
        where p1 = Normal(mean_1, exp(log_std_1)), p2 = Normal(mean_2, exp(log_std_2) ** 2).
        If mean_2 and log_std_2 are None values, we will use standard normal distribution.
    """
    if mean_2 is None:
        mean_2 = torch.zeros_like(mean_1)
    if log_std_2 is None:
        log_std_2 = torch.zeros_like(log_std_1)

    return (log_std_2 - log_std_1 +
            (torch.exp(2*log_std_1) + (mean_1 - mean_2) ** 2) / (2 * torch.exp(2*log_std_2)) - 0.5
           )


def get_normal_nll(x, mean, log_std):
    """
        returns the negative log likelihood log p(x),
        where p(x) = Normal(x | mean, exp(log_std) ** 2)
        (diagonal covariance matrix)
    """
    nll = -TD.Normal(mean, torch.exp(log_std)).log_prob(x)
    return nll


def kl_divergence_loss(mean, log_std):
    return ((mean**2 + (log_std**2).exp() - 1 - log_std**2) / 2).mean()


def train_epoch(
    model: object,
    train_data: pd.DataFrame,
    data_transformer: object,
    optimizer: object,
    batch_size: int,
    verbose: bool,
    loss_key: str = 'total_loss',
) -> defaultdict:
    model.train()

    stats = defaultdict(float)
    n = train_data.shape[0]
    for k in range(n // batch_size):
        i0, i1 = k * batch_size, (k + 1) * batch_size
        imputs = data_transformer.get_batch(train_data.iloc[i0:i1, :])   
        losses = model.loss(imputs)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()

        for k, v in losses.items():
            if k != 'recon_quality':
                stats[k] += v.item() * imputs[0].shape[0]
    for k in stats.keys():
        stats[k] /= n
    if verbose:
        print(f'train: {stats["recon_loss"]}, {stats["kl_loss"]}')
    return stats


def eval_model(model: object, 
               val_data: pd.DataFrame,
               data_transformer: object,
               batch_size: int,
               verbose: bool
              ) -> defaultdict:
    model.eval()
    stats = defaultdict(float)
    n = val_data.shape[0]
    recon_quality = []
    with torch.no_grad():
        
        for k in range(n // batch_size):
            i0, i1 = k * batch_size, (k + 1) * batch_size
            imputs = data_transformer.get_batch(val_data.iloc[i0:i1, :])   
            losses = model.loss(imputs)
            for k, v in losses.items():
                if k != 'recon_quality':
                    stats[k] += v.item() * imputs[0].shape[0]
                else:
                    recon_quality.append(v.cpu().detach().numpy())
        for k in stats.keys():
            stats[k] /= n
        if verbose:
            print(f'test: {stats["recon_loss"]}, {stats["kl_loss"]}')

    return stats, recon_quality


def train_model(
    model: object,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    data_transformer: object,
    batch_size: int, 
    epochs: int,
    lr: float,
    verbose: bool,
    use_tqdm: bool = False,
    device: torch.device = torch.device('cpu'),
    loss_key: str = "total_loss",
) -> Tuple[dict, dict]:
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)

    model = model.to(device)
    for epoch in forrange:
        model.train()
        train_loss = train_epoch(model, train_data, data_transformer, optimizer, batch_size, 
                                 verbose, loss_key=loss_key)
        test_loss, _ = eval_model(model, val_data, data_transformer, batch_size, verbose)

        if epoch > 0:
            for k in train_loss.keys():
                train_losses[k].append(train_loss[k])
                test_losses[k].append(test_loss[k])
    return dict(train_losses), dict(test_losses)



# ----- visualization

def plot_training_curves(
    train_losses: Dict[str, np.ndarray],
    test_losses: Dict[str, np.ndarray],
    logscale_y: bool = False,
    logscale_x: bool = False,
) -> None:
    n_train = len(train_losses[list(train_losses.keys())[0]])
    n_test = len(test_losses[list(train_losses.keys())[0]])
    x_train = np.linspace(0, n_test - 1, n_train)
    x_test = np.arange(n_test)

    plt.figure()
    for key, value in train_losses.items():
        plt.plot(x_train, value, label=key + "_train")

    for key, value in test_losses.items():
        plt.plot(x_test, value, label=key + "_test")

    if logscale_y:
        plt.semilogy()

    if logscale_x:
        plt.semilogx()

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()