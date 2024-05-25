# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import os
import logging
import os.path as op

LARGEST_Parquet_SIZE = 500_000

# LARGEST_Parquet_SIZE = 10_000


def create_lineidx(filein, idxout):
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as Parquetin, open(idxout_tmp, 'w') as Parquetout:
        fsize = os.fstat(Parquetin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            Parquetout.write(str(fpos)+"\n")
            Parquetin.readline()
            fpos = Parquetin.tell()
    os.rename(idxout_tmp, idxout)


def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return ''.join(result)


class ParquetFile(object):
    def __init__(self, Parquet_file, generate_lineidx=False):
        self.Parquet_file = Parquet_file
        self.lineidx = os.path.splitext(Parquet_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None
        self._table = pq.read_table(self.Parquet_file)
        self.df = self._table.to_pandas()  # 转换为 DataFrame 方便处理
        self.pid = None
        if not os.path.isfile(self.lineidx) and generate_lineidx:
            create_lineidx(self.Parquet_file, self.lineidx)

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "ParquetFile(Parquet_file='{}')".format(self.Parquet_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        assert len(
            self.df) <= LARGEST_Parquet_SIZE, f'Do not support ParquetFile larger than {LARGEST_Parquet_SIZE} yet'
        return len(self.df)

    def seek(self, idx):
        self._ensure_Parquet_opened()
        self._ensure_lineidx_loaded()
        try:
            row = self.df.iloc[idx]
        except:
            logging.info('{}-{}'.format(self.Parquet_file, idx))
            raise
        return row.tolist()

    def seek_first_column(self, idx):
        self._ensure_Parquet_opened()
        self._ensure_lineidx_loaded()
        row = self.df.iloc[idx]
        return row.iloc[0]

    def get_key(self, idx):
        return self.seek_first_column(idx)

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logging.debug('loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_Parquet_opened(self):
        if self._fp is None:
            self._fp = pq.ParquetFile(self.Parquet_file)
            self.pid = os.getpid()

        if self.pid != os.getpid():
            # logging.info('re-open {} because the process id changed'.format(self.Parquet_file))
            self._fp = pq.ParquetFile(self.Parquet_file)
            self.pid = os.getpid()