from mxnet.test_utils import *
import argparse


parser = argparse.ArgumentParser(description="Benchmark kvstore push sparse matrices",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-omp-threads', type=int, default=1, help='number of omp threads to set in MXNet')
parser.add_argument('--num-devs', type=int, default=4, help='number of devices')
parser.add_argument('--num-rows', type=int, default=50000, help='number of rows of the matrix')
parser.add_argument('--num-cols', type=int, default=50, help='number of columns of the matrix')
parser.add_argument('--dev-type', type=str, default='cpu', help='device type')
parser.add_argument('--stype', type=str, default='row_sparse', help='sparse type')
parser.add_argument('--key', type=int, default=1, help='key of the ndarray in kvstore')
parser.add_argument('--density', type=float, default=0.1, help='sparse ndarray density')


def push_dispatch(serial_push, key, vals, devs, shape, stype, density, num_devs):
    os.environ['MXNET_KVSTORE_SERIAL_PUSH'] = serial_push
    kv = mx.kv.create()
    kv.init(key, mx.nd.zeros(shape=shape, storage_type=stype))
    expected_sum = np.zeros(shape)
    for v in vals:
        expected_sum += v.asnumpy()
    out = [rand_ndarray(shape, stype, density).copyto(devs[i]) for i in range(num_devs)]
    kv.push(key, vals)
    kv.pull(key, out=out)
    result_sum = np.zeros(shape)
    for v in out:
        result_sum += v.asnumpy()
    assert_almost_equal(result_sum, expected_sum * num_devs, atol=0.0001, rtol=0.001)


def benchmark_sparse_aggregator(stype, shape, density, key, dev_type, num_devs):
    devs = [mx.Context(dev_type, i) for i in range(num_devs)]
    vals = [rand_ndarray(shape, stype, density).copyto(devs[i]) for i in range(num_devs)]
    push_dispatch('1', key, vals, devs, shape, stype, density, num_devs)
    push_dispatch('0', key, vals, devs, shape, stype, density, num_devs)


if __name__ == '__main__':
    args = parser.parse_args()
    benchmark_sparse_aggregator(args.stype, (args.num_rows, args.num_cols), args.density, args.key, args.dev_type,
                                args.num_devs)