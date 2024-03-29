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

import torch


class SerializationTool(object):
    @staticmethod
    def serialize_model_gradients(model: torch.nn.Module) -> torch.Tensor:
        """_summary_

        Args:
            model (torch.nn.Module): _description_

        Returns:
            torch.Tensor: _description_
        """
        gradients = [param.grad.data.view(-1) for param in model.parameters()]
        m_gradients = torch.cat(gradients)
        m_gradients = m_gradients.cpu()
        
        return m_gradients

    @staticmethod
    def deserialize_model_gradients(model: torch.nn.Module, gradients: torch.Tensor):
        idx = 0
        for parameter in model.parameters():
            layer_size = parameter.grad.numel()
            shape = parameter.grad.shape

            parameter.grad.data[:] = gradients[idx:idx+layer_size].view(shape)[:]
            idx += layer_size

    @staticmethod
    def serialize_model(model: torch.nn.Module) -> torch.Tensor:
        """Unfold model parameters
        
        Unfold every layer of model, concate all of tensors into one.
        Return a `torch.Tensor` with shape (size, ).

        Args:
            model (torch.nn.Module): model to serialize.
        """

        parameters = [param.data.view(-1).cpu() for param in model.parameters()]
        m_parameters = torch.cat(parameters)
        # m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def deserialize_model(model: torch.nn.Module,
                          serialized_parameters: torch.Tensor,
                          mode="copy"):
        """Assigns serialized parameters to model.parameters.
        This is done by iterating through ``model.parameters()`` and assigning the relevant params in ``grad_update``.
        NOTE: this function manipulates ``model.parameters``.

        Args:
            model (torch.nn.Module): model to deserialize.
            serialized_parameters (torch.Tensor): serialized model parameters.
            mode (str): deserialize mode. "copy" or "add".
        """
        total_size = serialized_parameters.size()[0]
        # print('total:',total_size)
        current_index = 0  # keep track of where to read from grad_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            # print('numel',numel)
            size = parameter.data.size()
            # print('size:',size)
            # print('current_index',current_index)
            # print(parameter.data)
            if mode == "copy":
                parameter.data = torch.rand(size)
                # print(size)
                current_index_add = current_index + numel
                # if current_index_add>total_size:
                #     break
                # print(current_index_add)
                # print(serialized_parameters[current_index:current_index_add].view(size).size())
                parameter.data.copy_(
                    serialized_parameters[current_index:current_index_add].view(size))
            
                
            elif mode == "add":
                parameter.data.add_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                    .format(mode))
            current_index += numel
