from trainer import Trainer
from config import get_parser
from data_loader import get_loader

def main():

    config = get_parser()

    dataloader = get_loader(config.data_dir, config.subjects, config.patch_size, config.mode, \
                            config.batch_size, config.num_workers, config.train_percentage)

    trainer = Trainer(dataloader, config)
    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'test':
        trainer.test()
    elif config.mode == 'train_test':
        trainer.train_test()
    else:
        raise ValueError

if __name__ == "__main__":
    main()
