
# def maybe_download_and_extract():
#     """Download and extract the tarball from Alex's website."""
#     dest_directory = FLAGS.data_dir
#     if not os.path.exists(dest_directory):
#         os.makedirs(dest_directory)
#     filename = DATA_URL.split('/')[-1]
#     filepath = os.path.join(dest_directory, filename)
#     if not os.path.exists(filepath):
#         def _progress(count, block_size, total_size):
#             sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
#                                                              float(count * block_size) / float(total_size) * 100.0))
#             sys.stdout.flush()
#         filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#         print()
#         statinfo = os.stat(filepath)
#         print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
#     extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
#     if not os.path.exists(extracted_dir_path):
#         tarfile.open(filepath, 'r:gz').extractall(dest_directory)
import os

import sys
import urllib.request
from shutil import copy2
from zipfile import ZipFile

REQUIRED_FILES = [
    'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
    'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
    'http://nlp.stanford.edu/data/glove.840B.300d.zip -O files/word-vectors.zip',
    'http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip',
    'http://nlp.stanford.edu/software/stanford-english-corenlp-2017-06-09-models.jar',
]


def prepare_data(data_dir: str):
    for url in REQUIRED_FILES:
        file_path = maybe_download(url, data_dir)

        if file_path.endswith('.zip'):
            resource_name = file_path.split('/')[-1].split('.')[0]
            extract_path = os.path.join(data_dir, resource_name)

            if not os.path.exists(extract_path):
                with ZipFile(file_path) as zip_file:
                    print('extracting {} ...'.format(file_path))
                    zip_file.extractall(extract_path)

    setup_parser(data_dir)


def maybe_download(url: str, dest_dir: str) -> str:
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    filename = url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    return filepath


def setup_parser(data_dir: str) -> None:
    parser_bin_dir = os.path.join(data_dir, 'stanford-corenlp-full-2017-06-09/stanford-corenlp-full-2017-06-09/bin')
    if not os.path.exists(parser_bin_dir):
        os.makedirs(parser_bin_dir)

    copy2(os.path.join(data_dir, 'javanlp.sh'), os.path.join(parser_bin_dir, 'javanlp.sh'))