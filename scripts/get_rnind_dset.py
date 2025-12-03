import argparse
from pathlib import Path
from tqdm import tqdm
from pyDataverse.api import NativeApi, DataAccessApi
from pyDataverse.models import Dataverse


def main():
    args_parser = argparse.ArgumentParser(
        description='Download Raw NIND dataset from https://dataverse.uclouvain.be/dataset.xhtml?persistentId=doi:10.14428/DVN/DEQCIM')
    args_parser.add_argument('--base_url', '-u', type=str, help='dataset base url',
                             default=r'https://dataverse.uclouvain.be')
    args_parser.add_argument('--doi', '-d', type=str, help='DOI of dataset',
                             default=r'doi:10.14428/DVN/DEQCIM')
    args_parser.add_argument('--save_dir', '-s', type=str, help='Path where to save',
                             default=r'D:/Projects/datasets/RNIND')
    args = args_parser.parse_args()

    save_dir_f = Path(args.save_dir)
    save_dir_f.mkdir(parents=True, exist_ok=True)

    api = NativeApi(args.base_url)
    data_api = DataAccessApi(args.base_url)
    dataset = api.get_dataset(args.doi)

    files_list = dataset.json()['data']['latestVersion']['files']

    for file in tqdm(files_list):
        filename = file["dataFile"]["filename"]
        file_id = file["dataFile"]["persistentId"]
        response = data_api.get_datafile(file_id)
        file_dist = save_dir_f / filename
        with open(file_dist.resolve(), "wb") as f:
            f.write(response.content)

if __name__ == '__main__':
    main()
