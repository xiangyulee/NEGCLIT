# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.multiprocessing import Process

from .network import DistNetwork

class NetworkManager(Process):
    """Abstract class.

    Args:
        network (DistNetwork): object to manage torch.distributed network communication.
    """
    def __init__(self, network: DistNetwork):
        super(NetworkManager, self).__init__()
        self._network = network

    def run(self,args):
        """
        Main Process:

          1. Initialization stage.
          2. FL communication stage.
          3. Shutdown stage. Close network connection.
        """
        print('network setup......')
        self.setup()
        print('network mainloop......')
        self.main_loop(args)
        print('network shutdown......')
        self.shutdown()

    def setup(self):
        """Initialize network connection and necessary setups.
        
        At first, ``self._network.init_network_connection()`` is required to be called.

        Overwrite this method to implement system setup message communication procedure.
        """
        self._network.init_network_connection()

    def main_loop(self,args):
        """Define the actions of communication stage."""
        raise NotImplementedError()

    def shutdown(self):
        """Shutdown stage.

        Close the network connection in the end.
        """
        self._network.close_network_connection()
