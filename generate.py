from __future__ import division

import argparse
import math

import torch
from s2s.generator import DialogGenerator

parser = argparse.ArgumentParser(description='generate.py')
parser.add_argument('-model', default="./data/stc/checkpoints/seq2seq_s2t__acc_17.33_ppl_105.29_e1.pt",
                    help='Path to model .pt file')
parser.add_argument('-t2s_model', default="./data/stc/checkpoints/seq2seq_t2s__acc_21.20_ppl_279.12_e2.pt")

parser.add_argument('-src', default="./data/stc/train/src.train.txt",
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-src_img_dir', default="",
                    help='Source image directory')
parser.add_argument('-trg', default="./data/stc/train/trg.train.txt",
                    help='True target sequence (optional)')
parser.add_argument('-output', default='./data/stc/test/pred.e3.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size', type=int, default=200,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=50,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=50,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', default=0,
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")

parser.add_argument('-verbose', default=0,
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=200,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def add_one(f):
    for line in f:
        yield line
    yield None


def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    generator = DialogGenerator(opt)

    out_fw = open(opt.output, 'w', encoding='utf-8')

    pred_score_total, pred_words_total, gold_score_total, gold_words_total = 0, 0, 0, 0

    src_batch, trg_batch = [], []

    count = 0

    trg_fr = open(opt.trg, 'r', encoding='utf-8') if opt.trg else None

    if opt.dump_beam != "":
        import json

    for line in add_one(open(opt.src, 'r', encoding='utf-8')):
        if line is not None:
            src_tokens = line.split()
            src_batch += [src_tokens]
            if trg_fr:
                trg_tokens = trg_fr.readline().split() if trg_fr else None
                trg_batch += [trg_tokens]

            if len(src_batch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(src_batch) == 0:
                break

        pred_batch, pred_score = generator.generate_conversation(src_batch, trg_batch)
        pred_score_total += sum(score[0] for score in pred_score)
        pred_words_total += sum(len(x[0]) for x in pred_batch)
        if trg_fr is not None:
            gold_words_total += sum(len(x) for x in trg_batch)

        for b in range(len(pred_batch)):
            count += 1
            for n in range(opt.n_best):
                out_fw.write("SENT %d: %s\t[%.4f]\t%s\n" % (
                    count,
                    ' '.join(src_batch[b]),
                    pred_score[b][n],
                    (" ".join(pred_batch[b][n]).replace("<unk>", ""))))

            # outF.write(" ".join(predBatch[b][0]) + '\n')
            out_fw.flush()
            if count % 10 == 0:
                print('generate {} lines over!'.format(count))

            if opt.verbose:
                src_sent = ' '.join(src_batch[b])
                if generator.trg_vocab.lower:
                    src_sent = src_sent.lower()
                print('SENT %d: %s' % (count, src_sent))
                print('PRED %d: %s' % (count, " ".join(pred_batch[b][0])))
                print("PRED SCORE: %.4f" % pred_score[b][0])

                if opt.n_best > 1:
                    print('\nBEST HYP:')
                    for n in range(opt.n_best):
                        print("[%.4f] %s" % (pred_score[b][n],
                                             " ".join(pred_batch[b][n])))
                print('')

        src_batch, trg_batch = [], []

    report_score('PRED', pred_score_total, pred_words_total)
    if trg_fr:
        report_score('GOLD', gold_score_total, gold_words_total)

    if trg_fr:
        trg_fr.close()

    if opt.dump_beam:
        json.dump(generator.beam_accum, open(opt.dump_beam, 'w'))

if __name__ == "__main__":
    main()
