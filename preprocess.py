import argparse

from utils.vocab import *
from utils.tree import *

parser = argparse.ArgumentParser(description='preprocess.py')

# **Pre-process Options**
parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-train_src', default="./data/stc/train/src.train.txt",
                    help="Path to the training source data")

parser.add_argument('-train_trg', default="./data/stc/train/trg.train.txt",
                    help="Path to the training target data")

parser.add_argument('-valid_src', default="./data/stc/test/src.test.txt",
                    help="Path to the validation source data")

parser.add_argument('-valid_trg', default="./data/stc/test/trg.test.txt",
                    help="Path to the validation target data")

parser.add_argument('-save_data', default="./data/stc/",
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=10000,
                    help="Size of the source vocabulary")
parser.add_argument('-trg_vocab_size', type=int, default=10000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-trg_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-src_seq_length', type=int, default=50,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-trg_seq_length', type=int, default=50,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-trg_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

parser.add_argument('-shuffle', type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed', type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)


def make_vocabulary(filename, size):
    vocab = Vocabulary([Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD],
                       lower=opt.lower)

    with open(filename, 'r', encoding='utf-8') as fr:
        for sent in fr:
            for word in sent.split():
                vocab.add(word)

    original_size = vocab.size
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size, original_size))

    return vocab


def init_vocabulary(name, data_file, vocab_file, vocab_size):
    vocab = None
    if vocab_file is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocab_file + '\'...')
        vocab = Vocabulary()
        vocab.load_file(vocab_file)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        vocab = make_vocabulary(data_file, vocab_size)

    print()
    return vocab


def save_vocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.write_file(file)


def filter_sentence(sent):
    import re
    sent = re.sub("[\s+\.\/_,$%^*(+\"\']+|[+——_、~@#￥%……&*（）【】😁]+", "", sent)
    sent = sent.replace("")
    return sent


def make_data(src_file, trg_file, src_vocab, trg_vocab):
    src, trg = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (src_file, trg_file))
    src_fr = open(src_file, 'r', encoding='utf-8')
    trg_fr = open(trg_file, 'r', encoding='utf-8')

    while True:
        src_line = src_fr.readline()
        trg_line = trg_fr.readline()

        # normal end of file
        if src_line == "" and trg_line == "":
            break

        # source or target does not have same number of lines
        if src_line == "" or trg_line == "":
            print('WARNING: src and trg do not have the same # of sentences')
            break

        src_line = src_line.strip()
        trg_line = trg_line.strip()

        # source and/or target are empty
        if src_line == "" or trg_line == "":
            print('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        src_words = src_line.split()
        trg_words = trg_line.split()

        if len(src_words) <= opt.src_seq_length \
                and len(trg_words) <= opt.trg_seq_length:

            # Check truncation condition.
            if opt.src_seq_length_trunc != 0:
                src_words = src_words[:opt.src_seq_length_trunc]
            if opt.trg_seq_length_trunc != 0:
                trg_words = trg_words[:opt.trg_seq_length_trunc]

            # if opt.src_type == "text":
            src += [src_vocab.convert2idx(src_words,
                                          Constants.UNK_WORD)]

            trg += [trg_vocab.convert2idx(trg_words,
                                          Constants.UNK_WORD,
                                          Constants.BOS_WORD,
                                          Constants.EOS_WORD)]
            sizes += [len(src_words)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)
        if count % 100000 == 0:
            break

    src_fr.close()
    trg_fr.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        trg = [trg[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.IntTensor(sizes))
    src = [src[idx] for idx in perm]
    trg = [trg[idx] for idx in perm]

    print(('Prepared %d sentences ' +
           '(%d ignored due to length == 0 or src len > %d or trg len > %d)') %
          (len(src), ignored, opt.src_seq_length, opt.trg_seq_length))

    return src, trg


def main():
    vocabs = {}
    vocabs['src'] = init_vocabulary('source', opt.train_src, opt.src_vocab,
                                    opt.src_vocab_size)

    vocabs['trg'] = init_vocabulary('target', opt.train_trg, opt.trg_vocab,
                                    opt.trg_vocab_size)

    print('Preparing training ...')
    train = {}
    train['src'], train['trg'] = make_data(opt.train_src, opt.train_trg,
                                           vocabs['src'], vocabs['trg'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['trg'] = make_data(opt.valid_src, opt.valid_trg,
                                           vocabs['src'], vocabs['trg'])

    if opt.src_vocab is None:
        save_vocabulary('source', vocabs['src'], opt.save_data + '/vocab/src.dict')
    if opt.trg_vocab is None:
        save_vocabulary('target', vocabs['trg'], opt.save_data + '/vocab/trg.dict')

    print('Saving vocabs to \'' + opt.save_data + '/vocab/vocab.pt\'...')
    torch.save(vocabs, opt.save_data + '/vocab/vocab.pt')

    print('Saving train data to \'' + opt.save_data + '/train/train.pt\'...')
    torch.save(train, opt.save_data + '/train/train.pt')

    print('Saving valid data to \'' + opt.save_data + '/valid/valid.pt\'...')
    torch.save(valid, opt.save_data + '/valid/valid.pt')


if __name__ == "__main__":
    main()
    pass
