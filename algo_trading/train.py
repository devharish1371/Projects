import argparse
import os

from dcrnn_supervisor import DCRNNSupervisor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', required=True, help='CSV with columns Date,Ticker,Close (and others)')
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--model_dir', default='logs/checkpoints')
    args = parser.parse_args()

    sup = DCRNNSupervisor(
        data=dict(
            data_path=args.data_csv,
            seq_len=args.seq_len,
            horizon=args.horizon,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            price_column='Close',
            use_returns=True,
        ),
        model=dict(
            rnn_units=64,
            num_rnn_layers=2,
            filter_type='random_walk',
        ),
        train=dict(
            epochs=args.epochs,
            base_lr=args.base_lr,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            lr_steps=[int(args.epochs * 0.6), int(args.epochs * 0.8)],
            lr_decay_ratio=0.5,
            patience=10,
        ),
        log_level='INFO'
    )

    sup.train()


if __name__ == '__main__':
    main()


