import os
import os.path
import sys


if __name__ == '__main__':
    # Check out first.
    os.system('git checkout .')
    os.system('git pull origin master')
    os.system('ant -f HandleMissing.xml all')
    print('You may not run:')
    print('python startup.py /home/fatty/Code/ml_datasets_arff ../result.txt')

