import paramiko
import os
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
        sftp.put(local_path, target_path)

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


def sock_server_data(args):


    ssh = SSHConnection(host=args.ip, port=22, username=args.username, pwd=args.password) #'1.tcp.vip.cpolar.cn'
    # ssh = SSHConnection(host=args.ip, port=args.port, username='root', pwd='linux123')

    ssh.connect()
    # ssh.cmd('mkdir -p /data/model')
    for filename in os.listdir(args.save_server):
        if filename.spilt('.')[-1] == '.tar':
            print('uploading:',filename)
            ssh.upload(args.save_server + filename, args.save_client+ filename)
    ssh.close()
