fig, ax = plt.subplots()
ax.imshow(image)

rect = patches.Rectangle(
    (
        x * IMAGE_SIZE - width * IMAGE_SIZE / 2,
        y * IMAGE_SIZE - height * IMAGE_SIZE / 2,
    ),
    width * IMAGE_SIZE,
    height * IMAGE_SIZE,
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)

anchor_rect = patches.Rectangle(
    (
        (grid_x + 0.5) * IMAGE_SIZE / GRID_SIZES[grid_index]
        - anchors[anchor_index][0] * IMAGE_SIZE / 2,
        (grid_y + 0.5) * IMAGE_SIZE / GRID_SIZES[grid_index]
        - anchors[anchor_index][1] * IMAGE_SIZE / 2,
    ),
    anchors[anchor_index][0] * IMAGE_SIZE,
    anchors[anchor_index][1] * IMAGE_SIZE,
    linewidth=1,
    edgecolor="b",
    facecolor="none",
)
ax.add_patch(anchor_rect)
plt.show()
