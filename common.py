from dataclasses import dataclass

@dataclass
class ConvConfig:
    batch_size: int
    in_channels: int
    out_channels: int
    input_size: tuple[int, int]
    kernel_size: int
    
    def __iter__(self):
        return iter((
            self.batch_size,
            self.in_channels,
            self.out_channels,
            self.input_size,
            self.kernel_size
        ))