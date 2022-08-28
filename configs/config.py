work_dir = './test'
save_play_history = False

self_play_config = dict(
    generation=16,
    num_searchs=64,
    num_games=64,
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
    max_actions=65,
    eps=1e-6,
    momentum=0.1,
    pretarined_path=None
)

train_config = dict(
    epochs=8,
    batch_size=32,
    num_workers=4,
    base_lr=5e-3,
    min_lr=1e-4,
    value_loss_weight=1.0,
    policy_loss_weight=1.0,
    traindata_rate=0.8
)
