import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.preload_data import *
from src.utils import read_config, plot_training_metrics
from gridworld.gridworld import GridWorld
from src.constants import *
from src.model import TemporalPCN, HierarchicalPCN
from src.trainer import PCTrainer


if __name__ == "__main__":
    conf_pomdp_path = './gridworld/config/pomdp.yaml'
    conf_mdp_path = './gridworld/config/mdp.yaml'
    run_path = './gridworld/config/run.yaml'
    setup_path = './gridworld/config/free.yaml'
    option_path = './configs/options.yaml'
    setup = read_config(setup_path)
    options = read_config(option_path)
    options['weight_decay'] = WEIGHT_DECAY
    options['decay_step_size'] = DECAY_STEP_SIZE
    options['decay_rate'] = DECAY_RATE
    options['lambda_z'] = LAMBDA_Z
    options['lambda_z_init'] = LAMBDA_Z_INIT
    options['loss'] = "CE"

    for mode in ['mdp', 'pomdp']:
        print(f"Training model with mode: {mode}")
        options['mode'] = mode
        if options['mode'] == 'mdp':
            options['obs_size'] = 2*len(setup['room'][0][0])
            conf = read_config(conf_mdp_path)
        else:
            options['obs_size'] = len(np.unique(setup['room'])) + 1
            conf = read_config(conf_pomdp_path)
        if "start_position" in setup:
            ini_pos = setup.pop("start_position")
        else:
            ini_pos = (None, None)
        
        options['room'] = setup['room']
        options['conf'] = conf
        device = options['device']
        model = TemporalPCN(options).to(device)
        init_model = HierarchicalPCN(options).to(device)
        trainer = PCTrainer(model=model, init_model=init_model, options=options, env=GridWorld)
        trainer.train(ini_pos=ini_pos)

        # vs, obs, init_actv, seq = trainer.traj_gen.generate_traj_data(ini_pos=ini_pos, save=False)
        # preds_xs, _ = trainer.predict(vs, init_actv)
        # print(seq.shape)
        # print("Ground truth: ", seq[:, :, 0, 0])
        # print("Decoded truth: ", trainer.traj_gen.decode_trajectory(obs)[0])
        # print("Decoded preds: ", trainer.traj_gen.decode_trajectory(preds_xs)[0])
        if mode == 'mdp':
            plot_conf = {'new_fig': True, 'show': False}
        else:
            plot_conf = {'save_path': "figures", 'new_fig': False, 'fig': fig, 'axs': ax, 
                         'colors': ['#2ca02c'] * 4}
            
            print(type(fig))
        fig, ax = plot_training_metrics(list(range(options['n_epochs'])), 
                            trainer.acc, trainer.loss, 
                            trainer.energy, trainer.acc_eval,
                            config=options, 
                            **plot_conf, label=mode)
    

