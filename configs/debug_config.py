work_dir = './works/debug'
save_play_history = False

self_play_config = dict(
    generation=3,
    num_searchs=16,
    num_games=64,
    game='Reversi',
    init_dict=dict(size=6),
    random_play=8,
    c_puct=1.0,
    temperature=0.1,
    num_cpus=4
)

model_config = dict(
    stem_config=dict(
        in_channels=2,
        out_dim=64,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0)
    ),

    block_config=dict(
        dim=64,
        eps=1e-6,
        momentum=0.1,
        se=True,
        act_fn='relu'
    ),
    depth=4,
    action_space=37,
    pretarined_path=None
)

train_config = dict(
    epochs=2,
    batch_size=64,
    num_workers=1,
    base_lr=5e-3,
    lr_gamma=0.5,
    value_loss_weight=1.0,
    policy_loss_weight=4.0,
    traindata_rate=0.8,
    hflip=False,
    vflip=False,
    device='cuda',
    use_amp=True
)