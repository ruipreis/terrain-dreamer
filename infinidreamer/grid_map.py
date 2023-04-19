import cv2
import numpy as np
import torch

class GridMap:
    def __init__(self, grid_size, spacing, clustered_images, deepfill_model, device):
        self.grid_size = grid_size
        self.spacing = spacing
        self.clustered_images = clustered_images
        self.deepfill_model = deepfill_model
        self.image_size = 256
        self.device = device

    def _get_image_coordinates(self, row, col):
        start_y = row * (self.image_size + self.spacing)
        end_y = start_y + self.image_size
        start_x = col * (self.image_size + self.spacing)
        end_x = start_x + self.image_size
        return start_y, end_y, start_x, end_x

    def _initialize_grid_map_and_mask(self, num_rows, num_cols):
        # Create grid_map and mask with the same dimensions
        height = num_rows * (self.image_size + self.spacing) - self.spacing
        width = num_cols * (self.image_size + self.spacing) - self.spacing

        grid_map = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)

        # Set mask values for the spacings between the images
        for row in range(num_rows):
            for col in range(num_cols):
                mask = self._update_mask(mask, row, col, num_rows, num_cols)

        return grid_map, mask

    def _update_mask(self, mask, row, col, num_rows, num_cols):
        if row < num_rows - 1:
            y1 = (row + 1) * self.image_size + row * self.spacing
            y2 = y1 + self.spacing
            x1 = col * (self.image_size + self.spacing)
            x2 = x1 + self.image_size
            mask[y1:y2, x1:x2] = 255
        if col < num_cols - 1:
            y1 = row * (self.image_size + self.spacing)
            y2 = y1 + self.image_size
            x1 = (col + 1) * self.image_size + col * self.spacing
            x2 = x1 + self.spacing
            mask[y1:y2, x1:x2] = 255
        if row > 0 and col > 0:
            y1 = row * (self.image_size + self.spacing) - self.spacing
            y2 = row * (self.image_size + self.spacing)
            x1 = col * (self.image_size + self.spacing) - self.spacing
            x2 = col * (self.image_size + self.spacing)
            mask[y1:y2, x1:x2] = 255

        return mask

    def create_map(self):
        # Initialize map dimensions and grid
        num_images = sum([len(cluster) for cluster in self.clustered_images])
        num_cols = int(np.ceil(np.sqrt(num_images)))
        num_rows = int(np.ceil(num_images / num_cols))
        grid_map, mask = self._initialize_grid_map_and_mask(num_rows, num_cols)

        # Place images on the grid
        image_counter = 0
        for cluster_idx, cluster in enumerate(self.clustered_images):
            for img, _ in cluster:
                row, col = divmod(image_counter, num_cols)
                start_y, end_y, start_x, end_x = self._get_image_coordinates(row, col)
                img_np = (img * 255).astype(np.uint8)
                grid_map[start_y:end_y, start_x:end_x] = img_np
                mask[start_y:end_y, start_x:end_x] = 0
                image_counter += 1

        return grid_map, mask

    def _random_bbox(self, height, width, min_scale_factor, max_scale_factor):
        min_dim = min(height, width)
        max_dim = max(height, width)

        min_mask_size = int(min_scale_factor * min_dim)
        max_mask_size = int(max_scale_factor * min_dim)

        mask_height = np.random.randint(min_mask_size, max_mask_size)
        mask_width = np.random.randint(min_mask_size, max_mask_size)

        top = np.random.randint(0, height - mask_height)
        left = np.random.randint(0, width - mask_width)

        return top, left, mask_height, mask_width

    def apply_border_inpainting(self, grid_map, mask, step_size=160):
        window_size = 256
        stride = step_size if step_size is not None else window_size // 2
        height, width, _ = grid_map.shape
        num_windows_height = (height - window_size) // stride + 1
        num_windows_width = (width - window_size) // stride + 1
        total_windows = num_windows_height * num_windows_width

        for i in range(num_windows_height):
            for j in range(num_windows_width):
                print(
                    f"Processed {i * num_windows_width + j + 1}/{total_windows} windows"
                )
                y_start, x_start = i * stride, j * stride
                y_end, x_end = y_start + window_size, x_start + window_size

                window = grid_map[y_start:y_end, x_start:x_end].copy()
                window_mask = mask[y_start:y_end, x_start:x_end].copy()

                # Normalize the image tensor to the range [-1, 1]
                window_tensor = (
                    torch.tensor(window, dtype=torch.float32)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 127.5
                ) - 1.0
                # import pdb; pdb.set_trace()
                masked_window_tensor = window_tensor.clone()
                masked_window_tensor[:, :, (window_mask == 255)] = 0

                # Normalize the mask tensor to the range [0, 1] and expand dimensions to match the image tensor
                window_mask_tensor = (
                    torch.tensor(window_mask, dtype=torch.float32)
                ) / 255
                window_mask_tensor = window_mask_tensor.unsqueeze(0).unsqueeze(0)

                # Display the input image
                # input_image_np = ((masked_window_tensor.squeeze(0).permute(1, 2, 0).detach().numpy() + 1) * 127.5).astype(np.uint8)
                # input_image_bgr = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR)
                # cv2.imshow("Input image for inpainting model", input_image_bgr)
                # cv2.waitKey(1)

                # Display the input mask
                # input_mask_np = (window_mask_tensor.squeeze(0).squeeze(0).detach().numpy() * 255).astype(np.uint8)
                # input_mask_bgr = cv2.cvtColor(input_mask_np, cv2.COLOR_GRAY2BGR)
                # cv2.imshow("Input mask for inpainting model", input_mask_bgr)
                # cv2.waitKey(1)

                # Run the inpainting model
                _, output = self.deepfill_model(
                    masked_window_tensor.to(self.device),
                    window_mask_tensor.to(self.device),
                )

                # Convert the output tensor back to the range [0, 255]
                output_np = (
                    (output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() + 1)
                    * 127.5
                ).astype(np.uint8)
                mask_indices = np.where(window_mask == 255)
                grid_map[y_start:y_end, x_start:x_end][mask_indices] = output_np[
                    mask_indices
                ]

                # output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
                # cv2.imshow("Inpainted output_1", output_bgr)
                # cv2.waitKey(1)

                # grid_map_bgr = cv2.cvtColor(grid_map, cv2.COLOR_RGB2BGR)
                # cv2.imshow("Full grid so far", grid_map_bgr)
                # cv2.waitKey(1)

                print("window_tensor shape:", window_tensor.shape)
                print("window_mask_tensor shape:", window_mask_tensor.shape)
                print("masked_window_tensor shape:", masked_window_tensor.shape)
                print(
                    "window_tensor min:",
                    window_tensor.min(),
                    "max:",
                    window_tensor.max(),
                )
                print(
                    "window_mask_tensor min:",
                    window_mask_tensor.min(),
                    "max:",
                    window_mask_tensor.max(),
                )

        return grid_map

    def apply_random_inpainting_old(self, grid_map, num_masks=10, min_scale_factor=0.1, max_scale_factor=0.4):
        return self._inpaint_with_random_masks(grid_map, num_masks, min_scale_factor, max_scale_factor)

    def apply_random_inpainting_in_windows_old(self, grid_map, window_size=256, step=160, num_masks=20, min_scale_factor=0.1, max_scale_factor=0.4):
        height, width, _ = grid_map.shape

        for y in range(0, height - window_size + 1, step):
            for x in range(0, width - window_size + 1, step):
                window = grid_map[y : y + window_size, x : x + window_size]
                window = self._inpaint_with_random_masks(window, num_masks, min_scale_factor, max_scale_factor)
                grid_map[y : y + window_size, x : x + window_size] = window

        return grid_map

    def _inpaint_with_random_masks(self, grid_map, num_masks=10, min_scale_factor=0.1, max_scale_factor=0.4):
        height, width, _ = grid_map.shape

        for i in range(num_masks):
            print(f"Processing random mask {i + 1}/{num_masks}")

            # Generate a random mask
            top, left, mask_height, mask_width = self._random_bbox(height, width, min_scale_factor, max_scale_factor)
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[top : top + mask_height, left : left + mask_width] = 255

            # Normalize the image tensor to the range [-1, 1]
            grid_map_tensor = (torch.tensor(grid_map, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 127.5) - 1.0
            masked_grid_map_tensor = grid_map_tensor.clone()
            masked_grid_map_tensor[:, :, (mask == 255)] = 0

            # Normalize the mask tensor to the range [0, 1] and expand dimensions to match the image tensor
            mask_tensor = (torch.tensor(mask, dtype=torch.float32)) / 255
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

            # Run the inpainting model
            _, output = self.deepfill_model(masked_grid_map_tensor.to(self.device), mask_tensor.to(self.device))

            # Convert the output tensor back to the range [0, 255]
            output_np = ((output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() + 1) * 127.5).astype(np.uint8)
            mask_indices = np.where(mask == 255)
            grid_map[mask_indices] = output_np[mask_indices]

        return grid_map
    
    def apply_random_inpainting_in_windows(self, grid_map, window_size=256, step=160, num_iterations=15, min_scale_factor=0.1, max_scale_factor=0.4):
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            grid_map = self._apply_random_inpainting_in_windows_once(grid_map, window_size, step, min_scale_factor, max_scale_factor)
        return grid_map

    def _apply_random_inpainting_in_windows_once(self, grid_map, window_size=256, step=160, min_scale_factor=0.1, max_scale_factor=0.4):
        height, width, _ = grid_map.shape

        for y in range(0, height - window_size + 1, step):
            for x in range(0, width - window_size + 1, step):
                window = grid_map[y : y + window_size, x : x + window_size]
                window = self._inpaint_with_random_masks(window, 3, min_scale_factor, max_scale_factor)
                grid_map[y : y + window_size, x : x + window_size] = window

        return grid_map