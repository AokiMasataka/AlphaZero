work_dir = './works/reversi_6x6'
save_play_history = False

self_play_config = dict(
    generation=8,
    num_searchs=128,
    num_games=512,
    game='Reversi',
    init_dict=dict(size=6),
    random_play=8,
    c_puct=1.0,
    temperature=1.0
)

model_config = dict(
    in_channels=2,
    dim=128,
    depth=4,
    max_actions=37,
    eps=1e-6,
    momentum=0.1,
    pretarined_path=None
)

train_config = dict(
    epochs=4,
    batch_size=128,
    num_workers=2,
    base_lr=5e-3,
    lr_gamma=0.5,
    value_loss_weight=0.25,
    policy_loss_weight=1.0,
    traindata_rate=0.8,
    hflip=True,
    vflip=True
)
