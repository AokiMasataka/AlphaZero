work_dir = './works/test'
save_play_history = False

self_play_config = dict(
    generation=3,
    num_searchs=16,
    num_games=16,
    game='Reversi',
    init_dict=dict(size=6),
    random_play=8,
    c_puct=1.0,
    temperature=0.1,
    num_cpus=1
)

model_config = dict(
    in_channels=2,
    dim=64,
    depth=2,
    action_space=37,
    eps=1e-6,
    momentum=0.1,
    pretarined_path=None
)

train_config = dict(
    epochs=2,
    batch_size=128,
    num_workers=1,
    base_lr=5e-3,
    lr_gamma=0.5,
    value_loss_weight=1.0,
    policy_loss_weight=4.0,
    traindata_rate=0.8,
    hflip=True,
    vflip=True,
    devica='cuda'
)