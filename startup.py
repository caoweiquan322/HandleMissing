import os
import os.path
import sys


if __name__ == '__main__':
    # Prepare class path.
    base_dir = './lib'
    cp = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]
    cp.append('./out/production/HandleMissing')

    # The java parameter.
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        raise Exception('The arff data directory not specified.')

    # The run.
    if len(sys.argv) > 2:  # Maybe we want to redirect the ouput.
        log_file = sys.argv[2]
        os.system('java -cp %s com.fatty.ml.Main %s > %s' % (':'.join(cp), data_dir, log_file))
    else:
        os.system('java -cp %s com.fatty.ml.Main %s' % (':'.join(cp), data_dir))

