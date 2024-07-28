
import torch

def binary_tensor_to_number(tensor: torch.Tensor) -> int:
    if tensor.size(0) != 8:
        raise ValueError("Tensor must be of size 8.")
    
    byte = 0
    for bit_index in range(8):
        if tensor[bit_index].item() > 0.5:
            byte |= (1 << (7 - bit_index))
    
    number = int.from_bytes([byte], "little", signed=False)
    return number


def number_to_binary_tensor(number: int) -> torch.Tensor:
        bytes = number.to_bytes(8, "little", signed=False)
        tensor = torch.Tensor(8)
        byte = bytes[0]
        for i in range(8):
            tensor[i] = 1.0 if byte & (1 << 7 - i) > 0 else 0.0
        return tensor