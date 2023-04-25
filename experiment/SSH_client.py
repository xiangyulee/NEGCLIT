
import os
import paramiko

class SSHConnection(object):

    def __init__(self, host, port, username, pwd):
        self.host = host
        self.port = port

        self.username = username
        self.pwd = pwd
        self.__k = None

    def connect(self):
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.pwd)
        self.__transport = transport

    def close(self):
        self.__transport.close()

    def upload(self, local_path, target_path):
        sftp = paramiko.SFTPClient.from_transport(self.__transport)
        sftp.put(os.path.join(os.getcwd(),local_path), os.path.join(os.getcwd(),target_path))

    def download(self, remote_path, local_path):
        sftp = paramiko.SFTPClient.from_transport(self.__transport)
        sftp.get(remote_path, local_path)

    def cmd(self, command):
        ssh = paramiko.SSHClient()
        ssh._transport = self.__transport
        # 执行命令
        stdin, stdout, stderr = ssh.exec_command(command)
        # 获取命令结果
        result = stdout.read()
        print(str(result, encoding='utf-8'))
        return result


def sock_client_data(args,target,input):


    ssh = SSHConnection(host=args.ip, port=22, username=args.username, pwd=args.password) #'1.tcp.vip.cpolar.cn'
    # ssh = SSHConnection(host=args.ip, port=args.port, username='root', pwd='linux123')

    ssh.connect()
    # ssh.cmd('mkdir -p /data/model')
    
        
    for filename in os.listdir(args.save_client):
        if filename.split('.')[-1] == 'npy':
            print('client uploading:',filename)
            ssh.upload(args.save_client + filename, args.save_server+ filename) #path change
    ssh.close()

def sock_server_data(args):


    ssh = SSHConnection(host=args.ip, port=22, username=args.username, pwd=args.password) #'1.tcp.vip.cpolar.cn'
    # ssh = SSHConnection(host=args.ip, port=args.port, username='root', pwd='linux123')

    ssh.connect()
    # ssh.cmd('mkdir -p /data/model') #这里是建立save_client文件夹的好时机 #path change

    if not os.path.exists(args.save_server):
        os.mkdir(args.save_server)
    if not os.path.exists(args.save_client): #
        os.mkdir(args.save_client)

    for filename in os.listdir(args.save_server):
        # print(filename)
        if filename.find('best')!=-1: # change
            print('server uploading:',filename)
            # print(args.save_server + filename)
            # print(args.save_client)
            ssh.upload(args.save_server + filename, args.save_client+ filename)  # 这里的save_server 如果是相对地址 就无法读取到 #path change
    ssh.close()