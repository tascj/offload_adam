import torch


class PinnedMemoryManager:
    """PyTorch allocates PowerOf2Ceil(requested memory) on CPU pinned memory, which could lead to unexpected
    host memory usage when training LLMs. For example, requesting 84GB of memory will result in 128GB being allocated.
    This class mitigates this issue by allocating memory across multiple (power-of-2 sized) buckets.
    """

    def __init__(
        self,
        parameters,
        config,
        bucket_size=4 * (1024**3),
        verbose=0,
    ):
        """
        Args:
            parameters: Iterator from model.parameters()
            config: Dict[str, torch.dtype], config for each state, e.g. {'exp_avg': torch.bfloat16, 'exp_avg_sq': torch.bfloat16, 'compensation': torch.bfloat16}
            bucket_size: int, size of each bucket (default 4GB)
        """
        self.buckets = []  # List[torch.Tensor]
        self.states = {}  # param -> Dict[str, (bucket_idx, offset)]
        self.parameters = set()
        self.config = config
        self.bucket_size = bucket_size
        self.verbose = verbose

        parameters = list(parameters)
        # Check maximum parameter size
        max_element_size = max(dtype.itemsize for dtype in config.values())
        max_param_size = max(
            (
                param.numel() * max_element_size
                for param in parameters
                if param.requires_grad
            ),
            default=0,
        )
        if max_param_size > bucket_size:
            raise ValueError(
                f"Buffer size ({bucket_size}) must be greater than or equal to "
                f"the largest parameter size ({max_param_size})"
            )

        # Initialize the first bucket
        self.buckets.append(
            torch.zeros(bucket_size, dtype=torch.uint8, device="cpu", pin_memory=True)
        )
        current_bucket = 0
        current_offset = 0

        # Allocate space for each parameter
        for param in parameters:
            if not param.requires_grad:
                continue
            self.parameters.add(param)
            # assert (
            #     param.dtype == torch.bfloat16
            # ), "Only bfloat16 parameters are supported"

            states = {}
            num_param_elements = param.numel()

            # Allocate space for each state
            for name, dtype in config.items():
                num_state_bytes = num_param_elements * dtype.itemsize
                # If the current bucket does not have enough space, create a new bucket
                if current_offset + num_state_bytes > bucket_size:
                    if self.verbose > 0:
                        print(
                            f"Used {current_offset / (1024**3):.2f}GB of {bucket_size / (1024**3):.2f}GB"
                        )
                    self.buckets.append(
                        torch.zeros(
                            bucket_size,
                            dtype=torch.uint8,
                            device="cpu",
                            pin_memory=True,
                        )
                    )
                    current_bucket += 1
                    current_offset = 0

                states[name] = (current_bucket, current_offset)
                current_offset += num_state_bytes

            self.states[param] = states
        if self.verbose > 0:
            print(
                f"Used {current_offset / (1024**3):.2f}GB of {bucket_size / (1024**3):.2f}GB"
            )

    @property
    def total_memory_allocated(self):
        """Total memory allocated in GB"""
        total_bytes = sum(bucket.numel() for bucket in self.buckets)
        return total_bytes / (1024**3)

    @property
    def total_memory_requested(self):
        """Total memory requested in GB"""
        bytes_per_param = sum(dtype.itemsize for dtype in self.config.values())
        requested_bytes = (
            sum(param.numel() for param in self.parameters) * bytes_per_param
        )
        return requested_bytes / (1024**3)

    def get(self, param, state_name, param_chunk=None):
        """Get view of specified state for a parameter

        Args:
            param: Model parameter
            state_name: Name of the state

        Returns:
            torch.Tensor: View of bfloat16 type
        """
        assert state_name in self.config, f"State {state_name} was not allocated"
        bucket_idx, offset = self.states[param][state_name]
        dtype = self.config[state_name]
        num_elements = param.numel()
        num_bytes = num_elements * dtype.itemsize
        ret = (
            self.buckets[bucket_idx][offset : offset + num_bytes]
            .view(dtype)
            .view(param.shape)
        )
        if param_chunk is not None:
            ret = ret.view(-1)[param_chunk]
        return ret
