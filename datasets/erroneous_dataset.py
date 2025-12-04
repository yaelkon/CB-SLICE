"""
Utility functions for filtering erroneous examples from CBM predictions.
Erroneous examples are those where the CBM's task predictions don't match
ground truth labels.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List


class ErroneousExamplesDataset(Dataset):
    """
    A dataset wrapper that filters to only include examples where the CBM model
    makes erroneous task predictions (y_pred != y_true).
    """

    def __init__(self, original_dataset: Dataset, cbm_model, device: torch.device,
                 num_classes: int):
        """
        Initialize the erroneous examples dataset.

        Args:
            original_dataset: The original dataset to filter
            cbm_model: The pretrained CBM model
            device: Device to run the model on
            num_classes: Number of classes for the task
        """
        self.dataset_name = original_dataset.dataset_name
        self.stage = original_dataset.stage
        self.original_dataset = original_dataset
        self.cbm_model = cbm_model
        self.device = device
        self.num_classes = num_classes
        self.class_errors = torch.zeros(num_classes)

        # Find erroneous examples
        self.erroneous_indices = self._find_erroneous_examples()

        print(f"Found {len(self.erroneous_indices)} erroneous examples out of "
              f"{len(original_dataset)} total examples")
        print(f"Error rate: {len(self.erroneous_indices) / len(original_dataset):.2%}")

    def _find_erroneous_examples(self) -> List[int]:
        """
        Find indices of examples where CBM task predictions are erroneous.

        Returns:
            List of indices corresponding to erroneous examples
        """
        # Batch processing for efficiency, preserving correct dataset indices
        from torch.utils.data import DataLoader

        batch_size = 64  # You can adjust this batch size as needed
        erroneous_indices = []
        self.cbm_model.eval()
        
        with torch.no_grad():
            # Create a DataLoader for the original dataset (no shuffle, sequential)
            loader = DataLoader(
                self.original_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            # If the dataset supports __getitem__ with dicts, we assume it returns a dict with "features" and "labels"
            # We need to keep track of the original indices in the dataset for each batch
            for batch in loader:
                # Compute the indices in the original dataset for this batch
                batch_indices = batch["img_code"].tolist()

                x = batch["features"].to(self.device)
                y_true = batch["labels"]
                cbm_outputs = self.cbm_model(x, validation=True)
                y_pred_logits = cbm_outputs["y_pred_logits"]

                if self.num_classes == 2:
                    # Binary classification
                    y_pred = (torch.sigmoid(y_pred_logits) > 0.5).long().squeeze(-1)
                else:
                    # Multi-class classification
                    y_pred = torch.argmax(y_pred_logits, dim=-1)

                # Compare predictions to ground truth
                mismatches = (y_pred.cpu() != y_true)
                for i in range(self.num_classes):
                    self.class_errors[i] += mismatches[y_true == i].sum().item()
                # Get the indices in the original dataset for erroneous examples
                batch_erroneous_indices = [batch_indices[i] for i in torch.where(mismatches)[0].tolist()]
                erroneous_indices.extend(batch_erroneous_indices)

        return erroneous_indices

    def __len__(self) -> int:
        return len(self.erroneous_indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Get an erroneous example by index.

        Args:
            idx: Index in the erroneous examples list

        Returns:
            The example data
        """
        original_idx = self.erroneous_indices[idx]
        return self.original_dataset[original_idx]


def create_erroneous_dataloader(original_loader: DataLoader, cbm_model,
                               device: torch.device, num_classes: int) -> DataLoader:
    """
    Create a DataLoader that only contains erroneous examples from the CBM model.

    Args:
        original_loader: The original DataLoader
        cbm_model: The pretrained CBM model
        device: Device to run the model on
        num_classes: Number of classes for the task

    Returns:
        DataLoader containing only erroneous examples
    """
    # Create erroneous dataset
    erroneous_dataset = ErroneousExamplesDataset(
        original_dataset=original_loader.dataset,
        cbm_model=cbm_model,
        device=device,
        num_classes=num_classes
    )

    # Create new DataLoader with same parameters as original
    erroneous_loader = DataLoader(
        erroneous_dataset,
        batch_size=original_loader.batch_size,
        shuffle=True,
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory,
        drop_last=original_loader.drop_last,
        generator=(original_loader.generator
                   if hasattr(original_loader, 'generator')
                   else None),
    )

    return erroneous_loader


def get_erroneous_examples_info(loader: DataLoader, cbm_model,
                               device: torch.device, num_classes: int) -> dict:
    """
    Get information about erroneous examples without creating a new dataset.

    Args:
        loader: The DataLoader to analyze
        cbm_model: The pretrained CBM model
        device: Device to run the model on
        num_classes: Number of classes for the task

    Returns:
        Dictionary with error statistics
    """
    cbm_model.eval()
    total_examples = 0
    erroneous_examples = 0
    class_errors = torch.zeros(num_classes)

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y_true = batch["labels"].to(device)

            # Get CBM predictions
            cbm_outputs = cbm_model(x, validation=True)
            y_pred_logits = cbm_outputs["y_pred_logits"]

            # Convert logits to predictions
            if num_classes == 2:
                # Binary classification
                y_pred = (torch.sigmoid(y_pred_logits) > 0.5).long()
            else:
                # Multi-class classification
                y_pred = torch.argmax(y_pred_logits, dim=-1)

            # Count errors
            batch_size = x.shape[0]
            total_examples += batch_size

            # Check which examples are erroneous
            for i in range(batch_size):
                if y_pred[i].item() != y_true[i].item():
                    erroneous_examples += 1
                    # Count class-level errors (true class)
                    class_errors[y_true[i]] += 1

    return {
        'total_examples': total_examples,
        'erroneous_examples': erroneous_examples,
        'error_rate': (erroneous_examples / total_examples if total_examples > 0
                       else 0),
        'class_errors': class_errors.cpu().numpy(),
        'avg_errors_per_class': class_errors.mean().item()
    }