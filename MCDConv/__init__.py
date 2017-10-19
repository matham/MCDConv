
from glob import glob
from os.path import splitext
from distutils.version import StrictVersion
from re import compile, match, sub
import numpy as np

import nixio as nix
from neo.io.nixio import NixIO
from neo.core import (
    Block, ChannelIndex, AnalogSignal, Segment)
from quantities import uV, Hz

mcd_version_min = StrictVersion('2.6.0')
mcd_version_max = StrictVersion('2.6.15')
header_pat = compile(br'MC_DataTool binary conversion\r\n'
br'Version ([0-9\.]+)\r\n'
br'MC_REC file = .+\r\n'
br'Sample rate = ([0-9\.]+)\r\n'
br'ADC zero = ([0-9]+)\r\n'
br'(.+)\r\n'
br'Streams = (.+)\r\n'
br'EOH\r\n')
scale_pat = compile(br'(An|Di|El) = ([0-9\.]+).+')
streams_pat = compile(br'An_|Di_|El_')

channels = [
    'A1', 'A2', 'A3', 'D1', '21', '31', '41', '51', '61', '71', '12', '22',
    '32', '42', '52', '62', '72', '82', '13', '23', '33', '43', '53', '63',
    '73', '83', '14', '24', '34', '44', '54', '64', '74', '84', '15', '25',
    '35', '45', '55', '65', '75', '85', '16', '26', '36', '46', '56', '66',
    '76', '86', '17', '27', '37', '47', '57', '67', '77', '87', '28', '38',
    '48', '58', '68', '78']


def read_header(filename):
    with open(filename, 'rb') as fh:
        data = fh.read(1000)
    m = match(header_pat, data)
    if m is None:
        print("Didn't find header for {}".format(filename))
        return None, None
    vrsn, rate, offset, scales, streams = m.groups()

    version = StrictVersion(vrsn)
    if version < mcd_version_min:
        raise Exception('{} header version {} is less than the minimum {}'.
                        format(filename, vrsn, mcd_version_min))
    if version > mcd_version_max:
        raise Exception('{} header version {} is higher than the maximum {}'.
                        format(filename, vrsn, mcd_version_max))
    rate = float(rate)
    offset = int(offset)
    streams = sub(streams_pat, '', streams).split(';')
    matches = [match(scale_pat, group) for group in scales.split(';')]
    scales = {m.group(1): float(m.group(2)) for m in matches}
    config = {'rate': rate, 'signed': not offset, 'channels': streams}
    if 'An' in scales:
        config['analog_scale'] = scales['An']
    if 'El' in scales:
        config['electrode_scale'] = scales['El']
    return m.end(), config


def read_files(filenames, dtype, chunks, slice_size):
    for filename in filenames:
        print('Processing {}'.format(filename))
        offset, config = read_header(filename)
        with open(filename, 'rb') as fh:
            if offset is not None:
                fh.seek(offset)

            while True:
                data = fh.read(chunks)
                if not len(data):
                    break

                assert not (len(data) % slice_size)
                yield np.frombuffer(data, dtype=dtype)


def create_nix_file(
    file_pat, output=None, signed=False, rate=1, electrode_scale=0.0104,
    analog_scale=12.5122, channels=channels, chunks=256 * 1024 * 1024):
    '''The default resolution (i.e. voltage per bit) in uV '''
    filenames = sorted(glob(file_pat))

    dtype = np.int16 if signed else np.uint16
    N = len(channels)
    slice_size = N * 2
    chunks = chunks - chunks % slice_size  # round to equal blocks

    reader = read_files(filenames, dtype, chunks, slice_size)
    data = next(reader)

    if signed:
        af = lambda x: x.astype(np.float32) * analog_scale
        ef = lambda x: x.astype(np.float32) * electrode_scale
        df = lambda x: np.array(x, dtype=np.bool_)
    else:
        af = lambda x: (x.astype(np.float32) - 2 ** 15) * analog_scale
        ef = lambda x: (x.astype(np.float32) - 2 ** 15) * electrode_scale
        df = lambda x: np.array(x.astype(np.int32) - 2 ** 15, dtype=np.bool_)

    analogs = [(i, ch) for i, ch in enumerate(channels) if ch.startswith('A')]
    digitals = [(i, ch) for i, ch in enumerate(channels) if ch.startswith('D')]
    electrodes = [
        (i, ch) for i, ch in enumerate(channels) if not ch.startswith('D') and
        not ch.startswith('A')]

    groups = (('Analog', analogs, af, np.float32),
              ('Digital', digitals, df, np.bool_),
              ('Electrodes', electrodes, ef, np.float32))

    if output is None:
        output = '{}.h5'.format(splitext(filenames[0])[0])
    ofile = NixIO(output, mode='ow')

    blk = Block()
    for group_name, channels, f, dtype in groups:
        seg = Segment(name=group_name)
        for slice_idx, chan_name in channels:
            seg.analogsignals.append(AnalogSignal(
                f(data[slice_idx::N]), dtype=dtype, units=uV,
                sampling_rate=rate * Hz, name=chan_name))
        blk.segments.append(seg)

    ofile.write_block(blk)
    nix_file = ofile.nix_file
    nix_groups = nix_file.blocks[0].groups

    for data in reader:
        for k, (group_name, channels, f, _) in enumerate(groups):
            data_arrays = nix_groups[k].data_arrays
            for i, (slice_idx, _) in enumerate(channels):
                # nix_file._h5file.flush()
                data_arrays[i].append(f(data[slice_idx::N]))

    nix_file.close()
    return output

if __name__ == '__main__':
    fname = r'g:\slice1_0000.raw'
    fname_pat = r'g:\slice1_0000.raw'
    print(create_nix_file(fname_pat, **read_header(fname)[1]))
