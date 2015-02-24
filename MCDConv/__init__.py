
from glob import glob
from os.path import splitext
from distutils.version import StrictVersion
from re import compile, match, sub
import numpy as np

from neo.io.hdf5io import NeoHdf5IO
from neo.core import (
    Block, RecordingChannelGroup,  RecordingChannel, AnalogSignal)
from quantities import uV, Hz

mcd_version_min = StrictVersion('2.6.0')
mcd_version_max = StrictVersion('2.6.0')
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



def convert_raw_files(
    file_pat, output=None, chunks=256 * 1024 * 1024, signed=False,
    rate=1, electrode_scale=0.0104, analog_scale=12.5122, channels=channels):
    '''The default resolution (i.e. voltage per bit) in uV '''
    filenames = sorted(glob(file_pat))

    N = len(channels)
    dtype = np.int16 if signed else np.uint16
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
    slice_size = len(channels) * 2
    chunks = chunks - chunks % slice_size  # round to equal blocks

    if output is None:
        output = '{}.h5'.format(splitext(filenames[0])[0])
    ofile = NeoHdf5IO(output, array_dtype=None)

    blk = Block()
    for channels, name, ctype in (
        (analogs, 'Analog', np.float32), (digitals, 'Digital', np.bool_),
        (electrodes, 'Electrodes', np.float32)):
        group = RecordingChannelGroup(
            name=name, channel_indexes=np.array([v[0] for v in channels]),
            channel_names=np.array([v[1] for v in channels]))
        for i, name in channels:
            chan = RecordingChannel(index=i)
            group.recordingchannels.append(chan)
            chan.recordingchannelgroups.append(group)
            chan.analogsignals.append(AnalogSignal(
                np.array([], dtype=ctype), dtype=ctype, units=uV,
                sampling_rate=rate * Hz, name=name))
        blk.recordingchannelgroups.append(group)

    ofile.write_block(blk)
    hdf_groups = ofile._data.root.Block_0.recordingchannelgroups
    channel_paths = []
    for i, (channels, f) in enumerate(
        ((analogs, af), (digitals, df), (electrodes, ef))):
        hdf_channs = getattr(
            hdf_groups, 'RecordingChannelGroup_{}'.format(i)).recordingchannels
        for j, (slice_idx, _) in enumerate(channels):
            channel_paths.append((
                getattr(hdf_channs, 'RecordingChannel_{}'.format(j)
                        ).analogsignals.AnalogSignal_0.signal, slice_idx, f))

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
                arr = np.fromstring(data, dtype=dtype)
                for signal, i, f in channel_paths:
                    x = f(arr[i::N])
                    x.resize((len(x), 1))
                    signal.append(x)
    ofile.close()
    return output

if __name__ == '__main__':
    fname = r'F:\MattE\all_file_header_test.raw'
    fname_pat = r'F:\MattE\all_file_header_test*.raw'
    convert_raw_files(fname_pat, **read_header(fname)[1])
    fname_pat = r'F:\MattE\all_file_test*.raw'
    convert_raw_files(fname_pat)
