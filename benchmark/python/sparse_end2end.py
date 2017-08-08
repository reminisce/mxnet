from mxnet.test_utils import *
import time
import argparse
import os

parser = argparse.ArgumentParser(description="Run sparse linear regression " \
                                             "with distributed kvstore",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--profiler', type=int, default=0,
                    help='whether to use profiler')
parser.add_argument('--num-epoch', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=512,
                    help='number of examples per batch')
parser.add_argument('--num-batch', type=int, default=99999999,
                    help='number of batches per epoch')
parser.add_argument('--dummy-iter', type=int, default=0,
                    help='whether to use dummy iterator to exclude io cost')
parser.add_argument('--kvstore', type=str, default='local',
                    help='what kvstore to use [local, dist_sync, etc]')
parser.add_argument('--log-level', type=str, default='debug',
                    help='logging level [debug, info, error]')
parser.add_argument('--dataset', type=str, default='avazu',
                    help='what test dataset to use')
parser.add_argument('--num-gpu', type=int, default=0,
                    help='number of gpus to use. 0 means using cpu(0);'
                         'otherwise, use gpu(0),...,gpu(num_gpu-1)')
parser.add_argument('--output-dim', type=int, default=1,
                    help='number of columns of the forward output')


def get_libsvm_data(data_dir, data_name, url, data_origin_name):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists(data_name)):
        import urllib
        zippath = os.path.join(data_dir, data_origin_name)
        urllib.urlretrieve(url, zippath)
        os.system("bzip2 -d %r" % data_origin_name)
    os.chdir("..")


class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch

# testing dataset sources
avazu = {
    'data_name': 'avazu-app.t',
    'data_origin_name': 'avazu-app.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.t.bz2",
    'feature_dim': 1000000,
}

kdda = {
    'data_name': 'kdda.t',
    'data_origin_name': 'kdda.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2",
    'feature_dim': 20216830,
}

datasets = { 'kdda' : kdda, 'avazu' : avazu }


def get_sym(feature_dim):
     initializer = mx.initializer.Normal()
     x = mx.symbol.Variable("data", stype='csr')
     norm_init = mx.initializer.Normal(sigma=0.01)
     w = mx.symbol.Variable("w", shape=(feature_dim, args.output_dim), init=norm_init, stype='row_sparse')
     embed = mx.symbol.dot(x, w)
     y = mx.symbol.Variable("softmax_label")
     model = mx.symbol.SoftmaxOutput(data=embed, label=y, name="out")
     return model


if __name__ == '__main__':

    # arg parser
    args = parser.parse_args()
    num_epoch = args.num_epoch
    num_batch = args.num_batch
    kvstore = args.kvstore
    profiler = args.profiler > 0
    batch_size = args.batch_size
    dummy_iter = args.dummy_iter
    dataset = args.dataset
    log_level = args.log_level
    contexts = mx.context.cpu(0) if args.num_gpu < 1\
        else [mx.context.gpu(i) for i in range(args.num_gpu)]

    # create kvstore when there are gpus
    kv = mx.kvstore.create(kvstore) if args.num_gpu >= 1 else None
    rank = kv.rank if kv is not None else 0
    num_worker = kv.num_workers if kv is not None else 1

    # only print log for rank 0 worker
    import logging
    if rank != 0:
        log_level = logging.ERROR
    elif log_level == 'DEBUG':
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=log_level, format=head)

    # dataset
    assert(dataset in datasets), "unknown dataset " + dataset
    metadata = datasets[dataset]
    feature_dim = metadata['feature_dim']
    if logging:
        logging.debug('preparing data ... ')
    data_dir = os.path.join(os.getcwd(), 'data')
    path = os.path.join(data_dir, metadata['data_name'])
    if not os.path.exists(path):
        get_libsvm_data(data_dir, metadata['data_name'], metadata['url'],
                        metadata['data_origin_name'])
        assert os.path.exists(path)

    # data iterator
    train_data = mx.io.LibSVMIter(data_libsvm=path, data_shape=(feature_dim,),
                                  batch_size=batch_size, num_parts=num_worker,
                                  part_index=rank)
    if dummy_iter:
        train_data = DummyIter(train_data)

    # model
    model = get_sym(feature_dim)

    # module
    mod = mx.mod.Module(symbol=model, data_names=['data'],
                        label_names=['softmax_label'], context=contexts)
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    sgd = mx.optimizer.SGD(momentum=0.0, clip_gradient=5.0,
                           learning_rate=0.1, rescale_grad=1.0/batch_size/num_worker)
    mod.init_optimizer(optimizer=sgd, kvstore=kv)
    # use accuracy as the metric
    metric = mx.metric.create('MSE')

    index = mod._exec_group.param_names.index('w')
    # weight_array bound to executors of the contexts
    weight_array = mod._exec_group.param_arrays[index]

    # start profiler
    if profiler:
        import random
        name = 'profile_output_' + str(num_worker) + '.json'
        mx.profiler.profiler_set_config(mode='all', filename=name)
        mx.profiler.profiler_set_state('run')

    logging.debug('start training ...')
    start = time.time()
    data_iter = iter(train_data)
    for epoch in range(num_epoch):
        nbatch = 0
        end_of_batch = False
        data_iter.reset()
        metric.reset()
        next_batch = next(data_iter)
        while not end_of_batch:
            nbatch += 1
            batch = next_batch

            mod.forward_backward(batch)
            # update parameters
            mod.update()
            # if have kvstore, need to pull corresponding rows of
            # the weights to each context
            if kv is not None:
                # column indices (NDArray type) of the csr data
                # used as the row_idx of the weight row-sparse matrix
                # TODO(junwu):
                # the following two lines block, may need to precompute
                # them and cache them outside the for loop
                row_indices = batch.data[0].indices
                indptr = batch.data[0].indptr.asnumpy()
                row_idx_array = []
                for s in mod._exec_group.slices:
                    row_idx_array.append(row_indices[indptr[s.start]:indptr[s.stop]])
                kv.row_sparse_pull('w', weight_array, priority=-index, row_ids=row_idx_array)

            try:
                # pre fetch next batch
                next_batch = next(data_iter)
                if nbatch == num_batch:
                    raise StopIteration
            except StopIteration:
                end_of_batch = True
            # accumulate prediction accuracy
            mod.update_metric(metric, batch.label)
        logging.info('epoch %d, %s' % (epoch, metric.get()))
    if profiler:
        mx.profiler.profiler_set_state('stop')
    end = time.time()
    time_cost = end - start
    logging.info('num_worker = ' + str(num_worker) + ', time cost = ' + str(time_cost))
