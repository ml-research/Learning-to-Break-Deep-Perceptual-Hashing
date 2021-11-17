import csv
import os
from os.path import join
from typing import List
from datetime import datetime
from random import randint


class Logger():
    def __init__(self, experiment_name: str, header: List, output_dir: str = './logs'):
        """
        Simple logger to log tabular experimental results as csv.

        Args:
            experiment_name (str): Name of the experiment.
            header (List): Header of the table.
            output_dir (str, optional): Where to save the file. Defaults to './logs'.
        """
        self.experiment_name = experiment_name
        self.header = header
        self.data = []
        self.output_dir = output_dir
        self.output_file = experiment_name + '.csv'

        try:
            os.mkdir(self.output_dir)
        except:
            print(f'Folder {output_dir} already exists.')

    def add_line(self, data: List):
        """
        Add new data row. Format should be consistent with the specified header.

        Args:
            data (List): DAta to log.
        """
        self.data.append(data)

    def finish_logging(self):
        """
        Finishing logging and write results in csv-file.
        """
        file = join(self.output_dir, self.output_file)
        if os.path.exists(file):
            print(f'File {file} already exists.')
            file = join(self.output_dir, str(
                randint(0, 10000)) + self.output_file)
            print(f'Created {file} instead to store results.')

        with open(file, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now()])
            writer.writerow(self.header)
            writer.writerows(self.data)
        print(f'Finishing logging. Results were written in {file}.')
