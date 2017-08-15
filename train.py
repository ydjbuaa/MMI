# -*- coding:utf-8 -*-
from __future__ import division

import time
from torch import cuda
from s2s.models import *
from utils.dataset import *
from utils.io_utils import *
from utils.optimizer import Optim

config_path = './config.json'
configs = load_configs(config_path)
print_configs(configs)

if torch.cuda.is_available() and not configs['gpus']:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if configs['gpus']:
    cuda.set_device(configs['gpus'][0])


def loss_criterion(vocab_size):
    weight = torch.ones(vocab_size)
    weight[Constants.PAD] = 0
    crit = nn.CrossEntropyLoss(weight)
    if configs['gpus']:
        crit.cuda()
    return crit


def memory_efficient_loss(outputs, targets, generator, crit, eval_flag=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval_flag), volatile=eval_flag)

    outputs_split = torch.split(outputs, configs['max_generator_batches'])
    targets_split = torch.split(targets, configs['max_generator_batches'])
    for i, (out_t, trg_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = generator(out_t)
        loss_t = crit(scores_t, trg_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(trg_t.data) \
            .masked_select(trg_t.ne(Constants.PAD).data) \
            .sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        if not eval_flag:
            loss_t.backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss / len(outputs_split), grad_output, num_correct


def evaluate(model, criterion, data):
    total_loss = 0
    total_words = 0
    total_num_correct = 0

    model.eval()
    for i in range(len(data)):
        # exclude original indices
        src_batch, trg_batch = data[i][:-1]
        outputs = model(src_batch, trg_batch[:-1])
        # exclude <s> from targets
        targets = trg_batch[1:]
        loss, _, num_correct = memory_efficient_loss(
            outputs, targets, model.generator, criterion, eval_flag=True)
        total_loss += loss
        total_num_correct += num_correct
        total_words += targets.data.ne(Constants.PAD).sum()

    model.train()
    return total_loss / len(data), total_num_correct / total_words


def train_model(model, train_set, valid_set, vocabs, optim):
    print(model)
    model.train()

    # Define criterion of each GPU.
    criterion = loss_criterion(vocabs['trg'].size)

    start_time = time.time()

    def train_epoch(epoch):

        if configs['extra_shuffle']:
            train_set.shuffle()

        # Shuffle mini batch order.
        batch_order = torch.randperm(len(train_set))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_trg_words = 0, 0
        report_src_words, report_num_correct = 0, 0
        start = time.time()
        for i in range(len(train_set)):

            batch_idx = batch_order[i]
            # Exclude original indices.
            src_batch, trg_batch = train_set[batch_idx][:-1]

            model.zero_grad()
            outputs = model(src_batch, trg_batch[:-1])
            # Exclude <s> from targets.
            targets = trg_batch[1:]

            loss, grad_output, num_correct = memory_efficient_loss(
                outputs, targets, model.generator, criterion)

            outputs.backward(grad_output)

            # Update the parameters.
            optim.step()

            num_words = targets.data.ne(Constants.PAD).sum()
            report_loss += loss
            report_num_correct += num_correct
            report_trg_words += num_words
            report_src_words += src_batch[0].data.sum()
            total_loss += loss
            total_num_correct += num_correct
            total_words += num_words
            if i % configs['log_interval'] == -1 % configs['log_interval']:
                print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
                       "%3.0f src tok/s; %3.0f trg tok/s; %6.0f s elapsed") %
                      (epoch, i + 1, len(train_set),
                       report_num_correct / report_trg_words * 100,
                       math.exp(report_loss / configs['log_interval']),
                       report_src_words / (time.time() - start),
                       report_trg_words / (time.time() - start),
                       time.time() - start_time))

                report_loss, report_trg_words = 0, 0
                report_src_words, report_num_correct = 0, 0
                start = time.time()

        return total_loss / len(train_set), total_num_correct / total_words

    for epoch in range(configs['start_epoch'], configs['num_epochs'] + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss, train_acc = train_epoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)
        print('Train accuracy: %g' % (train_acc * 100))

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = evaluate(model, criterion, valid_set)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        print('Validation accuracy: %g' % (valid_acc * 100))

        #  (3) update the learning rate
        optim.update_learning_rate(valid_ppl, epoch)

        model_state_dict = (model.module.state_dict() if len(configs['gpus']) > 1
                            else model.state_dict())
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'vocabs': vocabs,
            'config': configs,
            'epoch': epoch,
            'optim': optim
        }
        path = configs['data_dir'] + configs['checkpoints']
        if configs['trg2src']:
            path = path + 't2s_'
        else:
            path = path + 's2t_'

        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (path, 100 * valid_acc, valid_ppl, epoch))


def main():
    print("Loading dataset ....")

    dataset = load_dataset(configs['data_dir'] + configs['train_path'],
                           configs['data_dir'] + configs['valid_path'],
                           configs['data_dir'] + configs['vocab_path'])

    if configs['trg2src']:
        train_src = dataset['train']['trg']
        train_trg = dataset['train']['src']

        valid_src = dataset['valid']['trg']
        valid_trg = dataset['valid']['src']

    else:
        train_src = dataset['train']['src']
        train_trg = dataset['train']['trg']

        valid_src = dataset['valid']['src']
        valid_trg = dataset['valid']['trg']

    dict_checkpoint = (configs['train_from'] if configs['train_from']
                       else configs['train_from_state_dict'])
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    if configs['model'] == 'Seq2Seq':
        train_set = Dataset(train_src,
                            train_trg,
                            configs['batch_size'],
                            cuda=True,
                            volatile=False, ret_limit=20000)

        valid_set = Dataset(valid_src,
                            valid_trg,
                            configs['batch_size'],
                            cuda=True,
                            volatile=True, ret_limit=1000)
    else:
        raise ValueError("No Such Model")

    vocabs = dataset['vocabs']
    configs['src_vocab_size'] = vocabs['src'].size
    configs['trg_vocab_size'] = vocabs['trg'].size

    print(' * vocabulary size. source = %d; target = %d' %
          (vocabs['src'].size, vocabs['trg'].size))
    print(" * training samples:\t{};\tbatches:\t{}".format(train_set.num_samples, train_set.num_batches))
    print(" * valid samples:\t{};\tbatches:\t{}".format(valid_set.num_samples, valid_set.num_batches))
    print(' * maximum batch size. %d' % configs['batch_size'])

    print('Building model...')

    if configs['model'] == 'Seq2Seq':
        model = Seq2SeqDialogModel(configs)
    else:
        raise ValueError("No such Model")

    if configs['train_from']:
        print('Loading model from checkpoint at %s' % configs['train_from'])
        chk_model = checkpoint['model']
        model.load_state_dict(chk_model.state_dict())
        configs['start_epoch'] = checkpoint['epoch'] + 1

    if configs['train_from_state_dict']:
        print('Loading model from checkpoint at %s'
              % configs['train_from_state_dict'])
        model.load_state_dict(checkpoint['model'])
        configs['start_epoch'] = checkpoint['epoch'] + 1

    if len(configs['gpus']) >= 1:
        model.cuda()
    else:
        model.cpu()

    if len(configs['gpus']) > 1:
        model = nn.DataParallel(model, device_ids=configs['gpus'], dim=1)

    if not configs['train_from_state_dict'] and not configs['train_from']:
        for p in model.parameters():
            p.data.uniform_(-configs['param_init'], configs['param_init'])

        optim = Optim(
            configs['optim'], configs['learning_rate'], configs['max_grad_norm'],
            lr_decay=configs['learning_rate_decay'],
            start_decay_at=configs['start_decay_at']
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    optim.set_parameters(model.parameters())

    if configs['train_from'] or configs['train_from_state_dict']:
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)

    train_model(model, train_set, valid_set, vocabs, optim)


if __name__ == "__main__":
    main()
