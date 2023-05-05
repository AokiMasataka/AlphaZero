work_dir = './works/reversi_6x6'
save_play_history = False

self_play_config = dict(
    generation=16,
    num_searchs=64,
    num_games=128,
    game='Reversi',
    init_dict=dict(size=6),
    random_play=8,
    c_puct=1.0,
    temperature=1.0,
    num_cpus=8
)

model_config = dict(
    stem_config=dict(
        in_channels=2,
        out_dim=96,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0)
    ),

    block_config=dict(
        dim=96,
        eps=1e-6,
        momentum=0.1,
        se=True,
        act_fn='relu'
    ),
    depth=8,
    action_space=37,
    pretarined_path=None
)

train_config = dict(
    epochs=6,
    batch_size=64,
    num_workers=8,
    base_lr=0.01,
    lr_gamma=1.0,
    value_loss_weight=1.0,
    policy_loss_weight=1.0,
    traindata_rate=0.8,
    hflip=False,
    vflip=False,
    device='cuda',
    use_amp=False
)