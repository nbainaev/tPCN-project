import numpy as np
from src.utils import read_config, random_pair_generator
from src.constants import *
import pickle as pkl
import random as rd
import torch
from src.data_utils.data_utils import GridWorldEncoder
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def cluster_latent_vectors(latent_vectors, n_clusters=10, visualize=False):
    """
    Кластеризация латентных векторов
    latent_vectors: массив размером (n_samples, latent_dim)
    """
    if latent_vectors.ndim == 3:
        latent_vectors = latent_vectors.view(-1, latent_vectors.shape[2]).numpy()
    # Для визуализации можно использовать t-SNE
    if visualize:
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vectors)
    # Кластеризация с помощью K-means
    model = KMeans(n_clusters=n_clusters, random_state=42)
    # model = GaussianMixture(
    #     n_components=n_clusters,           # Ваше известное число кластеров
    #     covariance_type='full',   # 'full', 'tied', 'diag', 'spherical'
    #     random_state=42,
    #     max_iter=10
    # )
    # model = Birch(
    #     n_clusters=n_clusters,           # Конечное число кластеров
    #     threshold=0.5,          # Радиус кластера
    #     branching_factor=5     # Максимальное число подкластеров
    # )
    cluster_labels = model.fit_predict(latent_vectors)
    if visualize:
        plt.figure(figsize=(16, 12))
        
        # Генерируем 49 различимых цветов
        colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 20 цветов из tab20
        colors = np.vstack([colors, plt.cm.tab20b(np.linspace(0, 1, 20))])  # +20 цветов
        colors = np.vstack([colors, plt.cm.Set3(np.linspace(0, 1, 12))])    # +12 цветов
        colors = colors[:n_clusters]  # Берем первые 49 цветов
    
        # Рисуем кластеры
        for i in range(n_clusters):
            mask = cluster_labels == i
            plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                    c=[colors[i]], 
                    label=f'Cluster {i}', 
                    s=50, alpha=0.7, edgecolors='w', linewidth=0.5)
        
        plt.title('t-SNE визуализация 49 кластеров', fontsize=16)
        plt.xlabel('t-SNE dimension 1', fontsize=12)
        plt.ylabel('t-SNE dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.tight_layout()
        plt.show()
    
    return cluster_labels, model

def build_transition_matrix(states, model, n_clusters, actions=None, mapping=None, cluster_labels=None):
    """
    Построение матрицы переходов между кластерами
    
    sequences: список последовательностей (state, action, next_state)
    cluster_labels: метки кластеров для всех состояний
    n_clusters: количество кластеров
    actions: список возможных действий (опционально)
    """
    cluster_labels = cluster_labels.reshape(states.shape[0], -1)
    # Инициализация матрицы переходов
    if actions is not None:
        # 3D матрица: [action, from_cluster, to_cluster]
        transition_matrix = np.zeros((actions.shape[2], n_clusters, n_clusters))
    else:
        # 2D матрица: [from_cluster, to_cluster]
        transition_matrix = np.zeros((n_clusters, n_clusters))
    
    # Подсчет переходов
    for j, sequence in enumerate(states):
        # labels = cluster_labels[j]
        labels = model.predict(sequence)
        for i in range(len(sequence) - 1):
            if mapping is not None:
                current_cluster = mapping[labels[i]]
                next_cluster = mapping[labels[i + 1]]
            else:
                current_cluster = labels[i]
                next_cluster = labels[i + 1]
            if actions is not None:
                action = actions[j, i]
                action_idx = np.argmax(action)
                transition_matrix[action_idx, current_cluster, next_cluster] += 1
            else:
                transition_matrix[current_cluster, next_cluster] += 1
    
    # Нормализация (преобразование в вероятности)
    row_sums = transition_matrix.sum(axis=2, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix /= row_sums
    
    return transition_matrix

def generate_true_transition_matrix(field):
    """
    Генерирует матрицу переходов размерностью (4, N, N) для поля с стенками.
    
    Args:
        field: 2D массив NxN, где числа <0 представляют стенки
    
    Returns:
        transition_matrix: матрица размером (4, N, N), где:
        - transition_matrix[a, s, s'] = P(s' | s, a)
        - действия: 0=вверх, 1=вправо, 2=вниз, 3=влево
    """
    n = len(field)
    total_states = n * n
    
    # Инициализируем матрицу переходов (4 действия × N состояний × N состояний)
    transition_matrix = np.zeros((4, total_states, total_states), dtype=float)
    
    # Векторы направлений для каждого действия
    direction_vectors = [
        (-1, 0),  # 0: вверх
        (0, 1),   # 1: вправо
        (1, 0),   # 2: вниз
        (0, -1)   # 3: влево
    ]
    
    for i in range(n):
        for j in range(n):
            current_state = i * n + j
            
            # Пропускаем стенки - из них нельзя никуда перейти
            if field[i][j] < 0:
                continue
            
            # Для каждого действия (направления)
            for action in range(4):
                dx, dy = direction_vectors[action]
                ni, nj = i + dx, j + dy
                
                # Проверяем, можно ли перейти
                if 0 <= ni < n and 0 <= nj < n:
                    # В пределах поля
                    if field[ni][nj] >= 0:
                        # Успешный переход в проходимую клетку
                        target_state = ni * n + nj
                        transition_matrix[action, current_state, target_state] = 1.0
                    else:
                        # Попытка перейти в стенку - остаемся на месте
                        transition_matrix[action, current_state, current_state] = 1.0
                else:
                    # Выход за границы поля - остаемся на месте
                    transition_matrix[action, current_state, current_state] = 1.0
    
    sums = transition_matrix.sum(axis=2, keepdims=True)
    sums[sums == 0] = 1.0
    transition_matrix /= sums
    return transition_matrix

def get_trajectory(field, init_pos, n_steps, mode='mdp'):
    available_actions = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
    len_room = field.shape[0]
    y, x = init_pos
    used_actions = list(range(4))
    actions = []
    observations = [field[y, x]]
    available_inds = list(range(len_room))
    step = 0
    true_positions = [y * len_room + x]
    while step < n_steps:
        actions.append(rd.choice(used_actions))
        dy, dx = available_actions.get(actions[-1])
        if x + dx not in available_inds or y + dy not in available_inds:
            used_actions.remove(actions[-1])
            obs = -1
            true_pos = y * len_room + x
        elif field[y + dy, x + dx] < 0:
            used_actions.remove(actions[-1])
            obs = field[y, x]
            true_pos = y * len_room + x
        else:
            used_actions = list(range(4))
            x, y = x + dx, y + dy
            obs = field[y, x]
            true_pos = y * len_room + x
        observations.append(obs)
        true_positions.append(true_pos)
        step += 1  
    if mode == 'mdp':
        return np.array(actions, dtype=np.int16), np.array(true_positions, np.int16), np.array(true_positions, np.int16)
    else:
        return np.array(actions, dtype=np.int16), np.array(observations, np.int16), np.array(true_positions, np.int16) 

def plot_transition_matrices(predicted_matrix, true_matrix=None, titles=None, figsize=(15, 6)):
    """
    Визуализация одной или двух матриц переходов
    
    Parameters:
    -----------
    predicted_matrix : np.array
        Предсказанная матрица переходов
    true_matrix : np.array, optional
        Истинная матрица переходов
    titles : list, optional
        Заголовки для графиков
    figsize : tuple, optional
        Размер фигуры
    """
    
    if titles is None:
        titles = ['Predicted Transition Matrix', 'True Transition Matrix']
    
    # Определяем, нужно ли строить subplots
    if true_matrix is None:
        # Строим только одну матрицу
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plot_single_matrix(ax, predicted_matrix, titles[0])
    else:
        # Строим две матрицы рядом
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        plot_single_matrix(ax1, predicted_matrix, titles[0])
        plot_single_matrix(ax2, true_matrix, titles[1])
        
        # Добавляем общий заголовок
        fig.suptitle('Transition Matrices Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_single_matrix(ax, matrix, title):
    """
    Визуализация одной матрицы на заданных осях
    """
    # Создаем heatmap
    im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    
    # Настройки осей
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Добавляем цветовую шкалу
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Сетка
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.grid(False)

def find_cluster_mapping(true_labels, cluster_labels):
    """
    Находит соответствие кластеров истинным состояниям через majority voting
    """
    # Создаем матрицу сопряженности
    cm = confusion_matrix(true_labels, cluster_labels)
    
    n_true_states = cm.shape[0]
    n_clusters = cm.shape[1]
    
    # Для каждого кластера находим наиболее частое истинное состояние
    cluster_to_true = {}
    true_to_cluster = {}
    
    for cluster in range(n_clusters):
        true_state = np.argmax(cm[:, cluster])
        cluster_to_true[cluster] = true_state
        if true_state not in true_to_cluster:
            true_to_cluster[true_state] = cluster
    
    return cluster_to_true, true_to_cluster, cm

def predict_online(init_model, model, vs, pc_outputs, init_actv, options):
    
    model.eval()
    init_model.eval()
    pred_zs = []

    with torch.no_grad():
        init_model.inference(options['test_inf_iters'], options['inf_lr'], init_actv.to(options['device']))
        prev_hidden = init_model.z.clone().detach()
        pred_zs.append(pred_hidden)
        for k in range(options['n_steps_gen']):
            p = pc_outputs[:, k].to(options['device'])
            v = vs[:, k].to(options['device'])

            pred_hidden = model.g(v, prev_hidden)
            model.inference(options['inf_iters'], options['inf_lr'], v, prev_hidden, p)
            
            # update the hidden state
            prev_hidden = model.z.clone().detach()
            pred_zs.append(pred_hidden)
        
        pred_zs = torch.stack(pred_zs, dim=1) # [batch_size, sequence_length, Ng]
        pred_xs = model.decode(pred_zs)
    return pred_xs, pred_zs

def create_transition_matrix(init_model, model, options):
    room = options['room']
    len_room = len(options['room'][0])
    actions, observations = [], []
    true_positions = []
    dir_encoder = GridWorldEncoder(
            categories=[0, 1, 2, 3], 
            mode='directions', 
            encoder=options['encoder'])
    obs_encoder = GridWorldEncoder(
        categories=sorted(list(np.unique(options['room'][0]))  + [-1]) \
            if options['mode'] == 'pomdp' else list(range(len(options['room'][0]) ** 2)),
        mode=options['mode'],
        collision_hint=options['conf']['collision_hint'],
        encoder=options['encoder']
    )
    for _ in range(options['n_episodes']):
            for i in range(len_room):
                for j in range(len_room):
                    init_pos = (i, j)
                    episode_actions, episode_observations, episode_true_positions = get_trajectory(
                        field=np.array(options['room'][0]), 
                        init_pos=init_pos, 
                        n_steps=options['n_steps_gen'], 
                        mode=options['mode'])
                    actions.append(dir_encoder.transform(episode_actions).astype(np.int32))
                    observations.append(obs_encoder.transform(episode_observations).astype(np.int32))
                    true_positions.append(episode_true_positions)
    observations = np.stack(observations, axis=0)
    actions = np.stack(actions, axis=0)
    true_positions = np.stack(true_positions, axis=0)
    observations_tensors = torch.tensor(observations, dtype=torch.float32)
    actions_tensors = torch.tensor(actions, dtype=torch.float32)
    init_actv = observations_tensors[:, 0, :]
    observations_tensors = observations_tensors[:, 1:, :]

    preds, latent_vectors = predict_online(init_model, model, actions_tensors, observations_tensors, init_actv, options)
    
    cluster_labels, model = cluster_latent_vectors(latent_vectors, 
                                                    n_clusters=len_room ** 2, 
                                                    visualize=options['visualize_clusters']
                                                )
    cluster_to_true, true_to_cluster, cm = find_cluster_mapping(true_positions.flatten(), cluster_labels)
    transition_matrix = build_transition_matrix(
                            states=latent_vectors.numpy().astype(np.float32), 
                            actions=actions,
                            cluster_labels=cluster_labels, 
                            n_clusters=len_room ** 2, 
                            model=model,
                            mapping=cluster_to_true)
    
    true_transition_matrix = generate_true_transition_matrix(field=np.array(options['room'][0]))
    plot_transition_matrices(transition_matrix.mean(axis=0), true_transition_matrix.mean(axis=0))

def main():
    option_path = './configs/options.yaml'
    options = read_config(option_path)
    options['weight_decay'] = WEIGHT_DECAY
    options['decay_step_size'] = DECAY_STEP_SIZE
    options['decay_rate'] = DECAY_RATE
    options['lambda_z'] = LAMBDA_Z
    options['lambda_z_init'] = LAMBDA_Z_INIT
    options['loss'] = "CE"
    conf_pomdp_path = './gridworld/config/pomdp.yaml'
    conf_mdp_path = './gridworld/config/mdp.yaml'
    setup_path = './gridworld/config/free.yaml'
    setup = read_config(setup_path)

    options['setup'] = setup
    options['room'] = options['setup']['room']
    options['ini_pos'] = 'random'
    options['collision_hint'] = True
    options['prediction_mode'] = 'online'
    options['mode'] = 'pomdp'
    if options['mode'] == 'mdp':
        options['obs_size'] = len(options['setup']['room'][0][0]) ** 2
        conf = read_config(conf_mdp_path)
    else:
        options['obs_size'] = len(np.unique(options['setup']['room'])) + 1
        conf = read_config(conf_pomdp_path)
    if options['train_with_ini_pos'] == 'random':
        options['train_with_ini_pos'] = random_pair_generator
    if options['validate_with_ini_pos'] == 'random':
        options['validate_with_ini_pos'] = random_pair_generator
    options['conf'] = conf

    if "start_position" in setup:
        ini_pos = setup.pop("start_position")
    else:
        ini_pos = (None, None)
    
    with open(f'tpcn_init_model{'_pomdp_1' if options['mode'] == 'pomdp' else '_mdp'}.pkl', 'rb') as f:
        init_model = pkl.load(f)

    with open(f'tpcn_model{'_pomdp_1' if options['mode'] == 'pomdp' else '_mdp'}.pkl', 'rb') as f:
        model = pkl.load(f)

    options['visualize_clusters'] = True
    options['ini_pos'] = ini_pos
    options['n_steps_gen'] = 300
    options['n_episodes'] = 1
    create_transition_matrix(init_model=init_model, model=model, options=options)

if __name__ == '__main__':
    main()