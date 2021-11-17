from PIL import Image
import os
import sys
from os import listdir
from os.path import isfile, join


def main():
    folder_path = sys.argv[1]
    output_folder_path = folder_path + '_png'

    try:
        os.mkdir(output_folder_path)
    except:
        if not os.listdir(output_folder_path):
            print('Folder {output_folder_path} already exists and is empty.')
        else:
            print(
                'Folder {output_folder_path} already exists and is not empty.')

    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    for file in files:
        file_path = join(folder_path, file)
        img = Image.open(file_path)
        output_filename = file.rsplit(sep='.', maxsplit=1)[0] + '.png'
        output_path = join(output_folder_path, output_filename)
        img.save(output_path, format='png', quality=100)


if __name__ == "__main__":
    main()
