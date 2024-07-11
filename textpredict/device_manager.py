import torch  # type: ignore

from textpredict.logger import get_logger

logger = get_logger(__name__)


class DeviceManager:
    global_device = "cpu"

    @staticmethod
    def set_device(device="cpu"):
        """
        Set the global device for the application.
        Args:
            device (str): The preferred device ('cpu' or 'cuda').
        Returns:
            str: The device that will be used ('cpu' or 'cuda').
        """
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("GPU is not available. Falling back to CPU.")
                device = "cpu"
            else:
                try:
                    torch.zeros(1).cuda()  # Test if CUDA is working
                except Exception as e:
                    logger.warning(f"Failed to use GPU: {e}. Falling back to CPU.")
                    device = "cpu"

        DeviceManager.global_device = device
        logger.info(f"Global device set to {device}")
        return DeviceManager.global_device

    @staticmethod
    def get_device():
        """
        Get the current global device setting.
        Returns:
            str: The current device ('cpu' or 'gpu').
        """
        return DeviceManager.global_device
