
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = '/home/share/SGY/NewNet/test/LEVIR'
        elif data_name == 'LEVIR-256':
            self.label_transform = "norm"
            self.root_dir = '/home/share/SGY/NewNet/test/LEVIR-256'
        elif data_name == 'DSIFN':
            self.label_transform = "norm"
            self.root_dir = '/home/share/SGY/NewNet/test/DSIFN'
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir = '/home/share/SGY/NewNet/test/WHU'
        elif data_name == 'OSCD-256':
            self.label_transform = "norm"
            self.root_dir = '/home/share/SGY/NewNet/test/OSCD256'
        elif data_name == 'GZ':
            self.label_transform = "norm"
            self.root_dir = '/home/share/SGY/NewNet/test/GZ'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='DSIFN')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

