config = {
 'expname': 'brandenburg_explicit',
 'logdir': 'logs/TanksandTemple',
 'device': 'cuda:0',

 'data_downsample': 1,
 'data_dirs': ['data/nerf_synthetic/lego'], # ['data/phototourism/brandenburg_gate']
 'contract': True,
 'ndc': False,
 'scene_bbox': [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]],
 'global_scale': [1., 1., 1.],
 'global_translation': [0., 0., 0.], # 4 2 1
 # Optimization settings
 'num_steps': 30001,
 'batch_size': 4096,
 'optim_type': 'adam',
 'scheduler_type': 'warmup_cosine',
 'lr': 0.01,
 'app_optim_lr': 0.01,
 'app_optim_n_epochs': 10,

 # Regularization
 'plane_tv_weight': 0.0002, # 0.0002
 'plane_tv_weight_proposal_net': 0.0002, # 0.0002
 'distortion_loss_weight': 0.001,
 'histogram_loss_weight': 1.0,

 # Training settings
 'save_every': 10000,
 'valid_every': 10000,
 'save_outputs': True,
 'train_fp16': True,

 # Raymarching settings
 'single_jitter': False,
 'num_samples': 48,
 # proposal sampling
 'num_proposal_samples': [256, 128],
 'num_proposal_iterations': 2,
 'use_same_proposal_network': False,
 'use_proposal_weight_anneal': True,
 'proposal_net_args_list': [
   {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [32, 32, 32]},
   {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [64, 64, 64]}
 ],

 # Model settings
 'overlap_split': False,
 'expert_idx': None,
 'multiscale_res': [1, 2, 4, 8],
 'density_activation': 'trunc_exp',
 'concat_features_across_scales': True,
 'linear_decoder': True,
 'linear_decoder_layers': 1,
#  'appearance_embedding_dim': 32,
 'grid_config': [{
  'grid_dimensions': 2,
  'input_coordinate_dim': 3,
  'output_coordinate_dim': 32,
  'resolution': [32, 32, 32]
 }],
}
