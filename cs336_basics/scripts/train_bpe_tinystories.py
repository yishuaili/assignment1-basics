from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils.io import save_vocab_and_merge
import cProfile
import wandb

# io
text_source = 'data/TinyStoriesV2-GPT4-train.txt'
output_vocab_path = 'data/out/tinystories_vocab.json'
output_merge_path = 'data/out/tinystories_merges.txt'

# text_source = 'data/TinyStoriesV2-GPT4-valid.txt'
# output_vocab_path = 'data/out/tinystories_valid_vocab.json'
# output_merge_path = 'data/out/tinystories_valid_merges.txt'
# wandb setup
wandb_name = 'cs336_basics'
wandb_run_name = 'train_bpe_tinystories'
wandb_logging = False
# args
special_tokens = ['<|endoftext|>']
vocab_size = 10*(10**3)
num_workers = 5
# for wandb logging
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

def main():
    # wandb logging
    if wandb_logging:
        wandb.init(project=wandb_name, name=wandb_run_name, config=config)

    print("I am here!!!")

    # Training BPE
    pr = cProfile.Profile()
    pr.enable()
    vocab, merges = train_bpe(text_source, vocab_size, special_tokens, progress_bar=True, num_workers=num_workers)
    pr.disable()

    # Print time taken in units of hours
    pr.print_stats(sort='time')

    # Serialize and save the vocab and merges
    save_vocab_and_merge(vocab, merges, output_vocab_path, output_merge_path)

if __name__ == '__main__':
    main()
