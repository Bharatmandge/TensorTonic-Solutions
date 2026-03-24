import math

def roi_pool(feature_map, rois, output_size):
    # Handle output_size
    if isinstance(output_size, int):
        H = W = output_size
    else:
        H, W = output_size

    # Handle single ROI
    if isinstance(rois[0], int):
        rois = [rois]

    pooled_outputs = []

    for roi in rois:
        x1, y1, x2, y2 = roi

        roi_h = y2 - y1
        roi_w = x2 - x1

        output = []

        for i in range(H):
            row = []
            for j in range(W):

                # ✅ FIXED BIN CALCULATION
                h_start = y1 + (i * roi_h) // H
                h_end   = y1 + ((i + 1) * roi_h) // H

                w_start = x1 + (j * roi_w) // W
                w_end   = x1 + ((j + 1) * roi_w) // W

                # Ensure at least 1 pixel
                if h_end <= h_start:
                    h_end = h_start + 1
                if w_end <= w_start:
                    w_end = w_start + 1

                # Clip boundaries
                h_end = min(h_end, len(feature_map))
                w_end = min(w_end, len(feature_map[0]))

                max_val = float('-inf')

                for h in range(h_start, h_end):
                    for w in range(w_start, w_end):
                        max_val = max(max_val, feature_map[h][w])

                row.append(max_val)

            output.append(row)

        pooled_outputs.append(output)

    return pooled_outputs