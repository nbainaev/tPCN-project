import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import time
import json
import shutil
import matplotlib as mpl
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from ruamel.yaml import YAML
from pathlib import Path
from scipy.interpolate import make_smoothing_spline
from scipy.signal import savgol_filter

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

class UniqueColorPicker:
    def __init__(self, palette_name="husl", colors=None):
        """
        Инициализация выбора цветов из палитры seaborn.
        
        :param palette_name: Имя палитры (по умолчанию "husl" — хорошая палитра для различимости)
        """
        self.palette_name = palette_name
        self.available_colors = colors if colors is not None else sns.color_palette(palette_name)
        self.colors = colors
        self.used_colors = set()
        
    def get_unique_color(self):
        """
        Возвращает случайный не повторяющийся цвет из палитры.
        Если все цвета использованы, сбрасывает счётчик и начинает заново.
        """
        if not self.available_colors:
            # Если цвета закончились, сбрасываем и начинаем заново
            self.available_colors = self.colors if self.colors is not None else sns.color_palette(self.palette_name)
            self.used_colors = set()
        
        # Выбираем случайный цвет из оставшихся
        color = self.available_colors[0]
        
        # Удаляем его из доступных и добавляем в использованные
        self.available_colors.remove(color)
        self.used_colors.add(tuple(color))  # Конвертируем в tuple, т.к. список нехешируемый
        
        return color

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

def save_yaml(data, path):
    """
    Сохраняет данные в YAML-файл по относительному пути.
    Требует, чтобы целевая папка уже существовала.
    
    Параметры:
        data (dict): Данные для сохранения
        path (str): Относительный путь (с расширением или без)
                   Пример: "configs/exp1" → сохранит в "configs/exp1.yaml"
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.allow_unicode = True


    if not path.endswith(('.yaml', '.yml')):
        path += '.yaml'

    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)

    print(f"Файл сохранён: {path}")


def random_pair_generator(room, size):
    """
    Генератор бесконечных случайных пар чисел (кортежей)
    
    Параметры:
        min_val (int): Минимальное значение (включительно)
        max_val (int): Максимальное значение (включительно)
        
    Возвращает:
        tuple: Кортеж вида (x, y) где x и y ∈ [min_val, max_val]
    """
    room = np.array(room)
    room = -(room[0] < 0).astype(int) - np.array(room[2]) 
    max_y = room.shape[0] - 1
    max_x = room.shape[1] - 1
    points = []
    for i in range(size):
        while True:
            point = [random.randint(0, max_x), random.randint(0, max_y)]
            if room[point[0], point[1]] >= 0:
                break
        points.append(point)
    return np.array(points).squeeze()

def create_folder_with_datetime(base_name):
    """
    Создает папку с именем `base_name_YYYY-MM-DD_HH-MM`.
    Если папка уже существует - удаляет её и создает новую.
    Возвращает путь к созданной папке.
    """
    # Формат даты без секунд
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = f"{base_name}_{current_time}"
    
    # Удаляем папку, если она существует
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    
    # Создаем новую папку
    os.makedirs(folder_name, exist_ok=True)
    print(f"Папка создана: {folder_name}")
    return folder_name

def moving_average(y, window_size=2):
    return np.convolve(y, np.ones(window_size)/window_size, mode='same')

def plot_training_metrics(episodes, accuracy, total_loss=None, energy=None, accuracy_eval=None,
                          accuracy_time=None, accuracy_time_eval=None,
                          config=None, new_fig=True, fig=None, axs=None,
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
    metrics = [accuracy, accuracy_eval, accuracy_time['acc'], accuracy_time_eval['acc']]
    labels = ['Training Accuracy', 'Accuracy Eval', 'Accuracy Time', 'Accuracy Time Eval']
    n_subplots = 0
    for i, metric in enumerate(metrics):
        if metric is None:
            metrics.pop(i)
            labels.pop(i)
        else:
            n_subplots += 1
    if new_fig:
        fig, axs = plt.subplots(n_subplots // 2, 2, figsize=figsize, 
                            facecolor='#f5f5f5' if style != 'dark_background' else '#2b2b2b',
                            constrained_layout=True)
    
        fig.suptitle(f'Training Metrics Analysis {title_name.upper()}', fontsize=14)
    
    for i, (metric, Label) in enumerate(zip(metrics, labels)):
        if 'time' in Label.lower():
            x = list(range(len(metric['mean'])))
        elif 'eval' in Label.lower():
            x = episodes[::config['eval_every']]
        else:
            x = episodes
        if 'loss' not in Label.lower():
            spline_mean = make_smoothing_spline(x, np.array(metric['mean']))  # lam регулирует степень сглаживания
            smoothed_mean = np.array(spline_mean(x))
            smoothed_mean = savgol_filter(smoothed_mean, window_length=5, polyorder=3)

            spline_std = make_smoothing_spline(x, np.array(metric['std']))
            smoothed_std = np.array(spline_std(x))
            smoothed_std = savgol_filter(smoothed_std, window_length=5, polyorder=3)
            axs[i // 2, i % 2].fill_between(x, 
                                np.maximum(smoothed_mean - smoothed_std, 0), 
                                np.minimum(smoothed_mean + smoothed_std, 1), 
                                color=colors[i], 
                                alpha=0.3,
                                )
        else:
            spline_mean = make_smoothing_spline(x, np.array(metric), lam=0.3)  # lam регулирует степень сглаживания
            smoothed_mean = np.array(spline_mean(x))
            smoothed_mean = savgol_filter(smoothed_mean, window_length=5, polyorder=3)
        axs[i // 2, i % 2].plot(x, smoothed_mean, 
                    linewidth=linewidth, 
                    color=colors[i],
                    label=Label if not label else label)
        axs[i // 2, i % 2].set_title(Label, fontsize=title_fontsize)
        axs[i // 2, i % 2].set_xlabel('Time step' if 'Time' in Label else 'Epoch', fontsize=10)
        axs[i // 2, i % 2].set_ylabel('Accuracy' if 'Accuracy' in Label else 'Loss', fontsize=10)
        axs[i // 2, i % 2].grid(alpha=grid_alpha)
        axs[i // 2, i % 2].legend(loc='lower right')
        
    
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