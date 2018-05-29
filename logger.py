import os

class Logger():
    def __init__(self, exp_dir='exp'):
        self.exp_dir = os.path.join('.', exp_dir)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        self.log_path = os.path.join(exp_dir, 'log.txt')
        self.log_file = open(self.log_path, 'w+')

    def __del__(self): 
        self.log_file.close()

    def log(self, line):
        self.log_file.write(line)
        self.log_file.write('\n')
        self.log_file.flush()
        print(line)

    
