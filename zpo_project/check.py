import argparse
import requests
import yaml

def check(args):
    config_file = args.config
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    student_id = config['config']['STUDENT_ID']
    print("student Id: ", student_id)
    distance_name = 'cosine'  # supported values are: manhattan, euclidean, cosine
    with open('results.pickle', 'rb') as file:
        predictions = file.read()

    response = requests.post(f'https://zpo.dpieczynski.pl/{student_id}', headers={'distance': distance_name},
                             data=predictions)
    if response.status_code == 200:
        print(response.json())
    else:
        print(response.status_code)
        print(response.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-c', '--config', action='store', default='config.yaml')
    args = parser.parse_args()
    check(args)
