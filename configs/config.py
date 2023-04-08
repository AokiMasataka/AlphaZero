work_dir = './works/reversi_6x6'
save_play_history = False

self_play_config = dict(
    generation=16,
    num_searchs=196,
    num_games=1024,
    game='Reversi',
    init_dict=dict(size=6),
    random_play=8,
    c_puct=1.0,
    temperature=0.8,
    num_cpus=8
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
    epochs=6,
    batch_size=128,
    num_workers=8,
    base_lr=1e-3,
    lr_gamma=0.75,
    value_loss_weight=1.0,
    policy_loss_weight=4.0,
    traindata_rate=0.8,
    hflip=True,
    vflip=True,
    device='cuda',
    use_amp=True
)