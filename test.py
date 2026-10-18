import logging
import torch
import torch.nn.functional as F
import main
from backdoor_attack import apply_pixel_patterns, pattern_indices
from device import device

logger = logging.getLogger("logger")


def add_pixel_pattern(ori_image, params, handle, pattern_index=None):
    indices = pattern_indices(params, pattern_index=pattern_index)
    return apply_pixel_patterns(ori_image, params, indices, handle.target)


def normal_test(model, data_loader, params, handle, poison=False, poison_pattern_index=None):
    model.eval()
    test_loss = 0
    correct = 0
    data_num = 0

    if (
        params.get("gpu_preload", False)
        and hasattr(handle, "gpu_test_data")
        and poison_pattern_index is None
    ):
        if poison:
            test_data = handle.get_gpu_test_data(poison=True, target_label=False)
        else:
            test_data = handle.get_gpu_test_data(poison=False)
        use_gpu_preload = True
    else:
        test_data = data_loader
        use_gpu_preload = False

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_data):
            # Preloaded poisoned batches already contain the patch.
            if poison is True and not use_gpu_preload:
                for i in range(len(target)):
                    data[i], target[i] = add_pixel_pattern(
                        data[i],
                        params,
                        handle,
                        pattern_index=poison_pattern_index,
                    )
            if not use_gpu_preload:
                data, target = data.to(device), target.to(device)
            log_probs = model(data)

            test_loss += F.cross_entropy(log_probs, target, reduction="sum").item()
            data_num += len(target)
            pred = log_probs.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= data_num
    accuracy = 100.00 * correct / data_num
    main.logger.info(
        "___Global_Test_Poison:{}  Average loss: {:.4f} Accuracy: {}/{} ({:.2f}%)".format(
            poison, test_loss, correct, data_num, accuracy
        )
    )
    return accuracy
