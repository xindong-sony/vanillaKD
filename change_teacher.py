import torch

def shuffle_except_max(input_matrix):
    # Get the indices of max values along rows
    max_indices = torch.argmax(input_matrix, dim=1)
   
    # Create a mask for max elements
    max_mask = torch.zeros_like(input_matrix, dtype=torch.bool)
    max_mask[torch.arange(max_indices.size(0)), max_indices] = True
   
    # Invert the mask for non-max elements
    non_max_mask = ~max_mask

    # Get the non-max elements and shuffle them
    non_max_elements = input_matrix[non_max_mask].view(input_matrix.size(0), -1)
    shuffled_indices = torch.stack([torch.randperm(input_matrix.size(1) - 1) for _ in range(input_matrix.size(0))])
    shuffled_non_max_elements = non_max_elements.gather(1, shuffled_indices.to(input_matrix.device))
   
    # Place the shuffled non-max elements back into a copy of the original matrix
    shuffled_matrix = input_matrix.clone()
    shuffled_matrix[non_max_mask] = shuffled_non_max_elements.flatten()

    return shuffled_matrix


if __name__ == '__main__':
    # Test the function
    input_matrix = torch.rand((3, 5))
    print(input_matrix)
    print(shuffle_except_max(input_matrix))