import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
import matplotlib as mpl
import torch.optim as optim
import torch.nn.functional as F
from ruamel.yaml import YAML
from pathlib import Path

class Softmax(nn.Module):
    def forward(self, inp):
        return torch.softmax(inp, dim=-1)

    def deriv(self, inp):
        # Compute the softmax output
        soft = self.forward(inp)
        # Initialize a tensor for the derivative, with the same shape as the softmax output
        s = soft.unsqueeze(-2)  # Add a dimension for broadcasting
        identity = torch.eye(s.size(-1)).unsqueeze(0).to(inp.device)  
        # The diagonal contains s_i * (1 - s_i) and off-diagonal s_i * (-s_j)
        deriv = identity * s - s * s.transpose(-1, -2) # shape (batch_size, N, N)
        return deriv

class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0

class ReLU(nn.Module):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        out = self(inp)
        out[out > 0] = 1.0
        return out

class Sigmoid(nn.Module):
    def forward(self, inp):
        return torch.sigmoid(inp)

    def deriv(self, inp):
        out = self(inp)
        return out * (1 - out)

class Binary(nn.Module):
    def forward(self, inp, threshold=0.):
        return torch.where(inp > threshold, 1., 0.)

    def deriv(self, inp):
        return torch.zeros((1,))

class Linear(nn.Module):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return torch.ones((1,)).to(inp.device)

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def generate_run_ID(options):
    ''' 
    Create a unique run ID from the most relevant
    parameters. Remaining parameters can be found in 
    params.npy file. 
    '''
    params = [
        'steps', str(options.sequence_length),
        'batch', str(options.batch_size),
        options.RNN_type,
        str(options.Ng),
        options.activation,
        'rf', str(options.place_cell_rf),
        'DoG', str(options.DoG),
        'periodic', str(options.periodic),
        'lr', str(options.learning_rate),
        'weight_decay', str(options.weight_decay),
        'data_source', options.data_source,
        ]
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')

    return run_ID

def lognormal_sampler(mean, std, num_samples):
    mu = np.log((mean**2) / np.sqrt(mean**2 + std**2))
    sigma = np.sqrt(np.log(1 + (std**2) / (mean**2)))
    return np.exp(np.random.normal(mu, sigma, num_samples))

def ce_loss(output, _target):
    pred = F.softmax(output, dim=-1)
    return -(_target * torch.log(pred)).sum(-1).mean()

def serialize_complex_val(val):
    """Custom method to serialize complex objects."""
    if isinstance(val, nn.Module):
        return {"__nn_module__": val.__class__.__name__}
    else:
        return val

def save_options_to_json(options, filename='options.json'):
    options_dict = {}
    for attr in dir(options):
        if not attr.startswith('__'):
            val = getattr(options, attr)
            options_dict[attr] = serialize_complex_val(val)
    
    with open(filename, 'w') as f:
        json.dump(options_dict, f, indent=4, default=str)

def read_config(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open('r') as config_io:
        yaml = YAML()
        return yaml.load(config_io)

def plot_training_metrics(episodes, accuracy, total_loss, energy, accuracy_eval=None, config=None, new_fig=True, fig=None, axs=None,
                         figsize=(10, 8), style='seaborn-v0_8', label='', show=True,
                         colors=None, linewidth=2.5, title_name='', fig_name=None,
                         grid_alpha=0.3, title_fontsize=12,
                         save_path=None):
    """
    Улучшенная визуализация метрик обучения с настройкой стилей
    
    Параметры:
        episodes: range/array - номера эпизодов
        accuracy: array - значения accuracy
        figsize: tuple - размер фигуры
        style: str - стиль matplotlib ('seaborn', 'ggplot', 'dark_background' etc.)
        colors: list - цвета графиков [accuracy, loss, loss_p, loss_g]
        linewidth: float - толщина линий
        grid_alpha: float - прозрачность сетки
        title_fontsize: int - размер шрифта заголовков
        save_path: str - путь для сохранения (None - не сохранять)
    """
    ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4']
    if colors is None:
        colors = ['#1f77b4'] * 4
    
    # Устанавливаем стиль
    plt.style.use(style)
    mpl.rcParams['font.family'] = 'DejaVu Sans'  # Шрифт с поддержкой кириллицы
    
    # Создаем фигуру с 4 субплогами
    if new_fig:
        fig, axs = plt.subplots(2, 2, figsize=figsize, 
                            facecolor='#f5f5f5' if style != 'dark_background' else '#2b2b2b',
                            constrained_layout=True)
    
        fig.suptitle(f'Training Metrics Analysis {title_name.upper()}', fontsize=14)
    

    axs[0, 0].plot(episodes, accuracy, color=colors[0], linewidth=linewidth, 
                  label='Accuracy' if not label else label)
    axs[0, 0].set_title('Model Accuracy', fontsize=title_fontsize)
    axs[0, 0].set_xlabel('Epochs', fontsize=10)
    axs[0, 0].set_ylabel('Accuracy', fontsize=10)
    axs[0, 0].grid(alpha=grid_alpha)
    axs[0, 0].legend(loc='lower right')
    axs[0, 1].plot(episodes, total_loss, color=colors[1], linewidth=linewidth, 
                  label='Obs Loss' if not label else label, linestyle='--')
    axs[0, 1].set_title('Obs Training Loss', fontsize=title_fontsize)
    axs[0, 1].set_xlabel('Epochs', fontsize=10)
    axs[0, 1].set_ylabel('Loss', fontsize=10)
    axs[0, 1].grid(alpha=grid_alpha)
    axs[0, 1].legend()

    axs[1, 0].plot(episodes, energy, color=colors[2], linewidth=linewidth,
                   label='Total loss' if not label else label)
    axs[1, 0].set_title('Total loss', fontsize=title_fontsize)
    axs[1, 0].set_xlabel('Epochs', fontsize=10)
    axs[1, 0].set_ylabel('Loss', fontsize=10)
    axs[1, 0].grid(alpha=grid_alpha)
    axs[1, 0].legend()
    
    if accuracy_eval is not None:
        axs[1, 1].plot(episodes[::config['eval_every']], accuracy_eval, color=colors[3], linewidth=linewidth,
                    label='Accuracy Eval' if not label else label, linestyle='-.')
        axs[1, 1].set_title('Accuracy Eval', fontsize=title_fontsize)
        axs[1, 1].set_xlabel('Epochs', fontsize=10)
        axs[1, 1].set_ylabel('Accuracy', fontsize=10)
        axs[1, 1].grid(alpha=grid_alpha)
        axs[1, 1].legend()
    
    # Настройка общего вида
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Сохранение если нужно
    if save_path:
        #name = f"mdp_mode - {config['mdp_mode']}, episodes - {config['episodes']}, bacth_size - {config['batch_size']}, steps - {config['steps']}, fixed - {config['is_fixed']}, id-{time.time() // 2}.jpg"
        name = f"epochs - {config['n_epochs']}, batch_size - {config['batch_size']}, n_steps - {config['n_steps']}, seq_len - {config['sequence_length']}.jpg" if not fig_name else fig_name
        plt.savefig(save_path + "/" + name, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    if show:
        plt.show()
    
    return fig, axs