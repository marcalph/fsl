# import matplotlib.pyplot as plt
# import torchvision

# imgs, targets = next(iter(train_data_loader))
# support_imgs, query_imgs, _, _ = split_batch(imgs, targets)
# support_grid = torchvision.utils.make_grid(support_imgs,
#                                            nrow=K_SHOT,
#                                            normalize=True,
#                                            pad_value=0.9)
# support_grid = support_grid.permute(1, 2, 0)
# query_grid = torchvision.utils.make_grid(query_imgs,
#                                          nrow=K_SHOT,
#                                          normalize=True,
#                                          pad_value=0.9)
# query_grid = query_grid.permute(1, 2, 0)

# fig, ax = plt.subplots(1, 2, figsize=(8, 5))
# ax[0].imshow(support_grid)
# ax[0].set_title("Support set")
# ax[0].axis("off")
# ax[1].imshow(query_grid)
# ax[1].set_title("Query set")
# ax[1].axis("off")
# fig.suptitle("Few Shot Batch", weight="bold")
# fig.show()
# import time
# time.sleep(15)
# plt.close()
